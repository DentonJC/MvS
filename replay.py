import torch
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ClassBalancedBuffer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from copy import deepcopy
from torchvision.models import mobilenet_v3_small
from avalanche.training.utils import cycle


import torch
import torchvision.transforms as T
from PIL import Image
import imagehash
import numpy as np

class ImageHashFeatures:
    def __init__(self, hash_func='phash'):
        """
        Args:
            hash_func (str): One of 'phash', 'ahash', 'dhash', 'whash', etc.
        """
        self.hash_func = getattr(imagehash, hash_func)
        self.to_pil = T.ToPILImage()

    def features(self, x):
        """
        Compute image hash features for a GPU or CPU batch of RGB images.

        Args:
            x (Tensor): A batch of images (B, 3, H, W) with values in [0, 1].

        Returns:
            Tensor: Float tensor of shape (B, N) with values in [0, 1].
        """
        device = x.device
        x_cpu = x.detach().cpu()  # move to CPU for PIL compatibility

        hash_vectors = []
        for img_tensor in x_cpu:
            img_pil = self.to_pil(img_tensor)
            img_hash = self.hash_func(img_pil)
            hash_array = np.array(img_hash.hash, dtype=np.float32).flatten()
            hash_vectors.append(hash_array)

        hash_tensor = torch.tensor(hash_vectors, dtype=torch.float32)
        return hash_tensor.to(device)  # move back to original device



class MIRReplay:
    def __init__(self, model, buffer, args, device):
        self.model = model
        self.buffer = buffer
        self.args = args
        self.device = device

    def get_grad_vector(self):
        grad_vector = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.view(-1))
        return torch.cat(grad_vector)

    def get_virtual_model(self, grad_vector):
        model_copy = deepcopy(self.model).to(self.device)
        pointer = 0
        with torch.no_grad():
            for param in model_copy.parameters():
                numel = param.data.numel()
                param.data -= self.args.lr * grad_vector[pointer:pointer + numel].view_as(param.data)
                pointer += numel
        return model_copy

    def compute_scores(self, dataset, method):
        base_method = method.replace('_min', '').replace('_max', '')
        loader = DataLoader(dataset, batch_size=self.args.subsample, shuffle=True, pin_memory=True)
        logits_list, labels_list, tasks_list, inputs = [], [], [], []
        with torch.no_grad():
            for x, y, t in loader:
                x = x.to(self.device, non_blocking=True)
                logits = self.model(x)
                logits_list.append(logits)
                labels_list.append(y.to(self.device))
                inputs.append(x)
                tasks_list.append(t)
                #if 'mir' in method:
                break


        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        inputs = torch.cat(inputs)
        tasks = torch.cat(tasks_list)
    
        if base_method == 'entropy':
            probs = F.softmax(logits, dim=1)
            scores = (probs * torch.log(probs + 1e-8)).sum(1)
        elif base_method == 'confidence':
            scores = logits.max(1)[0]
        elif base_method == 'margin':
            sorted_logits, _ = logits.sort(descending=True)
            scores = sorted_logits[:, 0] - sorted_logits[:, 1]
        elif base_method == 'bayesian':
            T = 10
            logits_mc = []
            for _ in range(T):
                logits_mc.append(F.softmax(self.model(inputs), dim=1))
            probs = torch.stack(logits_mc)
            mean_probs = probs.mean(0)
            entropy1 = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
            entropy2 = -torch.sum(probs * torch.log(probs + 1e-8), dim=2).mean(0)
            scores = entropy1 - entropy2
        elif base_method == 'mir':
            grad_vector = self.get_grad_vector()
            virtual_model = self.get_virtual_model(grad_vector)
            logits_pre = self.model(inputs)
            logits_post = virtual_model(inputs)
            scores = F.cross_entropy(logits_post, labels, reduction='none') - \
                     F.cross_entropy(logits_pre, labels, reduction='none')
        else:
            raise ValueError(f"Unknown method {method}")
    
        descending = method.endswith('_max') or base_method == 'mir'
        return inputs, labels, tasks, scores, descending


    def select(self, method, batch_size):
        dataset = self.buffer.buffer
        if len(dataset) < self.args.batch_size_mem:
            # Not enough samples, just return them all
            loader = DataLoader(dataset, batch_size=len(dataset))
            x, y, t = next(iter(loader))
            return x.to(self.device), y.to(self.device), t.to(self.device)

        if method == "random":
            # Original behavior: random samples
            loader = DataLoader(dataset, batch_size=self.args.batch_size_mem, shuffle=True)
            x, y, t = next(iter(loader))
            return x.to(self.device), y.to(self.device), t.to(self.device)
        
        inputs, labels, tasks, scores, descending = self.compute_scores(dataset, method)
        sorted_indices = torch.argsort(scores, descending=descending)
        if self.args.mode == 'old':
            selected = sorted_indices[:batch_size]
        else:
            probs = torch.linspace(1, 0.01, scores.shape[0])
            probs /= probs.sum()
            selected = torch.multinomial(probs, batch_size, replacement=False)
        selected = selected.cpu()
        return inputs[selected].to(self.device), labels[selected].to(self.device), tasks[selected].to(self.device)

    def kmeans_select(self, batch_size):
        dataset = self.buffer.buffer
        loader = DataLoader(dataset, batch_size=512, shuffle=False, pin_memory=True)
        all_embeddings, all_inputs, all_t, all_labels = [], [], [], []
    
        with torch.no_grad():
            for x, y, t in loader:
                x = x.to(self.device, non_blocking=True)
                emb = self.model.return_hidden(x).cpu().numpy()
                all_embeddings.append(emb)
                all_inputs.append(x.cpu())
                all_labels.append(y)
                all_t.append(t)
        
        try:
            embeddings = np.vstack(all_embeddings).squeeze(-1).squeeze(-1)
        except:
            embeddings = np.vstack(all_embeddings)
        bx = torch.cat(all_inputs)
        by = torch.cat(all_labels)
        bt = torch.cat(all_t)

        kmeans = KMeans(n_clusters=batch_size, random_state=0, n_init="auto").fit(embeddings)
        centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
    
        selected_indices = []
        for c in range(batch_size):
            indices = np.where(cluster_labels == c)[0]
            if len(indices) == 0:
                continue
            dists = np.linalg.norm(embeddings[indices] - centers[c], axis=1)
            selected = indices[np.argmin(dists)]
            selected_indices.append(selected)
    
        selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        return bx[selected_indices].to(self.device), by[selected_indices].to(self.device), bt[selected_indices].to(self.device)


    def coreset_select(self, batch_size):
        dataset = self.buffer.buffer
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        all_inputs, all_labels, all_t = [], [], []
        with torch.no_grad():
            for x, y, t in loader:
                x = x.to(self.device)
                emb = self.model.return_hidden(x).cpu().numpy()
                all_inputs.append((x.cpu(), emb))
                all_labels.append(y)
                all_t.append(t)

        bx = torch.cat([x for x, _ in all_inputs])
        by = torch.cat(all_labels)
        bt = torch.cat(all_t)
        embeddings = np.vstack([e for _, e in all_inputs])

        N = embeddings.shape[0]
        start_idx = np.random.randint(N)
        selected = [start_idx]
        remaining = list(set(range(N)) - set(selected))

        for _ in range(batch_size - 1):
            dists = np.min(
                np.linalg.norm(embeddings[remaining][:, None] - embeddings[selected], axis=2),
                axis=1
            )
            new_idx = remaining[np.argmax(dists)]
            selected.append(new_idx)
            remaining.remove(new_idx)

        selected_indices = torch.tensor(selected)
        return bx[selected_indices].to(self.device), by[selected_indices].to(self.device), bt[selected_indices].to(self.device)

    def select_replay_samples(self, method, batch_size):
        if any(m in method for m in ['mir', 'entropy', 'confidence', 'margin', 'bayesian', 'random']):
            return self.select(method, batch_size)
        elif method == 'kmeans':
            return self.kmeans_select(batch_size)
        elif method == 'coreset':
            return self.coreset_select(batch_size)
        else:
            raise ValueError(f"Unknown selection method: {method}")


class Replay(SupervisedTemplate):
    def __init__(
        self,
        model,
        optimizer,
        criterion=CrossEntropyLoss(),
        mem_size=200,
        batch_size_mem=None,
        train_mb_size=1,
        train_epochs=1,
        eval_mb_size=1,
        device="cpu",
        plugins=None,
        evaluator=default_evaluator,
        eval_every=-1,
        remove_current=False,
        peval_mode="epoch",
        selection_strategy="mir",
        steps='one',
        mode='old',
        args=None,
        ti=False,
        **kwargs
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
            **kwargs
        )

        self.batch_size_mem = batch_size_mem or train_mb_size
        self.mem_size = mem_size
        self.selection_strategy = selection_strategy
        self.steps = steps
        self.mode = mode
        self.args = args
        self.buffer = ClassBalancedBuffer(mem_size, adaptive_size=True)
        self.ti = ti
        self.replay_loader = None

        if args.external == "pretrained":
            rmodel = mobilenet_v3_small(num_classes=1000, weights='IMAGENET1K_V1')
            rmodel.to(device)
            rmodel.eval()
        elif args.external == "random":
            rmodel = mobilenet_v3_small(num_classes=1000, weights=None)
            rmodel.to(device)
            rmodel.eval()
        elif args.external == "hashes":
            rmodel = ImageHashFeatures.features()
        else:
            rmodel = model
            rmodel.to(device)
            rmodel.eval()
        
        self.rmodel = rmodel
            
        self.replay_selector = MIRReplay(rmodel, self.buffer, self.args, device)
        self.selected_replay_x = None
        self.selected_replay_y = None

    def _after_training_exp(self, **kwargs):
        self.buffer.post_adapt(self, self.experience)
        self.selected_replay_x = None
        self.selected_replay_y = None
        super()._after_training_exp(**kwargs)

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            self.optimizer.zero_grad()
            self._before_backward(**kwargs)
            self.loss = self._make_empty_loss()

            buffer_dataset = self.buffer.buffer
            
            if getattr(self, "steps", None) == 'reverse':
                # 1) update on one batch from memory (if available)
                if len(buffer_dataset) >= self.batch_size_mem:
                    self.replay_loader = cycle(
                        torch.utils.data.DataLoader(
                            buffer_dataset,
                            batch_size=self.batch_size_mem,
                            shuffle=True,
                            drop_last=True,
                            num_workers=kwargs.get("num_workers", 0),
                        )
                    )
                if self.replay_loader is not None:
                    mem_batch = next(self.replay_loader)
                    # unpack memory batch
                    if self.ti:
                        mem_x, mem_y, mem_t = mem_batch
                        mem_x, mem_y, mem_t = (
                            mem_x.to(self.device),
                            mem_y.to(self.device),
                            mem_t.to(self.device),
                        )
                    else:
                        mem_x, mem_y, _ = mem_batch
                        mem_x, mem_y = (
                            mem_x.to(self.device),
                            mem_y.to(self.device),
                        )
            
                    # zero grads, forward, backward, step
                    self.optimizer.zero_grad()
                    if self.ti:
                        mem_out = self.model(mem_x, mem_t)
                    else:
                        mem_out = self.model(mem_x)
                    mem_loss = CrossEntropyLoss()(mem_out, mem_y)
                    mem_loss.backward()
                    self.optimizer_step()
            
                # 2) now update on the *current* minibatch
                self.optimizer.zero_grad()
                if self.ti:
                    cur_out = self.model(self.mbatch[0], self.mbatch[2])
                else:
                    cur_out = self.model(self.mbatch[0])
                cur_loss = CrossEntropyLoss()(cur_out, self.mbatch[1])
                cur_loss.backward()
                self.optimizer_step()
            
                # hook into Avalanche’s post-iteration logic
                self._after_training_iteration(**kwargs)
                continue


            if self.steps == 'two':
                if self.ti:
                    self.mb_output = self.model(self.mbatch[0], self.mbatch[2])
                else:
                    self.mb_output = self.forward()
                self._after_forward(**kwargs)

                #self.loss += self.criterion()
                if self.ti:
                    loss_current = CrossEntropyLoss()(self.model(self.mbatch[0], self.mbatch[2]), self.mbatch[1])
                else:
                    loss_current = CrossEntropyLoss()(self.model(self.mbatch[0]), self.mbatch[1])
                loss_current.backward()
                #self.backward()

                if len(buffer_dataset) >= self.batch_size_mem:
                    replay_x, replay_y, replay_t = self.replay_selector.select_replay_samples(
                        self.selection_strategy, self.batch_size_mem
                    )
                    self.model.train()
                    self.selected_replay_x = replay_x
                    self.selected_replay_y = replay_y
                else:
                    self.selected_replay_x = None
                    self.selected_replay_y = None

                if self.selected_replay_x is not None:
                    #self.mbatch[0] = self.selected_replay_x
                    #self.mbatch[1] = self.selected_replay_y
                    #self.mb_output = self.forward()
                    #CrossEntropyLoss()(self.mb_output, self.mbatch[1]).backward()
                    #self._after_backward(**kwargs)
                    if not self.ti:
                        loss_replay = CrossEntropyLoss()(self.model(self.selected_replay_x), self.selected_replay_y)
                    else:
                        loss_replay = CrossEntropyLoss()(self.model(self.selected_replay_x, replay_t), self.selected_replay_y)
                    loss_replay.backward()
                    self._after_backward(**kwargs)

            else:
                if len(buffer_dataset) >= self.batch_size_mem and self.selection_strategy in ["mir"]:
                    # Step 1: Compute grad vector from current batch
                    self.optimizer.zero_grad()
                    if self.ti:
                        out = self.rmodel(self.mbatch[0], self.mbatch[2])
                    else:
                        out = self.rmodel(self.mbatch[0])
                    loss = CrossEntropyLoss()(out, self.mbatch[1])
                    loss.backward()  # this populates gradients for MIR
                    # Step 2: Use MIR to select replay samples
                    replay_x, replay_y, replay_t = self.replay_selector.select_replay_samples(
                        self.selection_strategy, self.batch_size_mem
                    )
                    self.model.train()
                    # Step 3: Zero grads before real backward
                    self.optimizer.zero_grad()
                    # Step 4: Merge current and replay minibatches
                    self.mbatch[0] = torch.cat([self.mbatch[0], replay_x], dim=0)
                    self.mbatch[1] = torch.cat([self.mbatch[1], replay_y], dim=0)
                    self.mbatch[2] = torch.cat([self.mbatch[2], replay_t], dim=0)
                elif len(buffer_dataset) >= self.batch_size_mem:
                    # for non-MIR selection
                    replay_x, replay_y, replay_t = self.replay_selector.select_replay_samples(
                        self.selection_strategy, self.batch_size_mem
                    )
                    self.mbatch[0] = torch.cat([self.mbatch[0], replay_x], dim=0)
                    self.mbatch[1] = torch.cat([self.mbatch[1], replay_y], dim=0)
                    self.mbatch[2] = torch.cat([self.mbatch[2], replay_t], dim=0)

                if self.ti:
                    self.mb_output = self.model(self.mbatch[0], self.mbatch[2])
                else:
                    self.mb_output = self.forward()
                self._after_forward(**kwargs)

                self.loss += self.criterion()
                self._before_backward(**kwargs)
                self.backward()
                self._after_backward(**kwargs)

            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
