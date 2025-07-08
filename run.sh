# adam

guild run main.py batch_size=[32] batch_size_mem=[32] model=[cnn] dataset=[cifar10] memory_size=[200,500,1000] lr=[0.001] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[adamw] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[cnnbn] dataset=[cifar10] memory_size=[200,500,1000] lr=[0.003] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[adamw] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[resnet18] dataset=[cifar10] memory_size=[200,500,1000] lr=[0.0005] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[adamw] --yes --label base &
wait

guild run main.py batch_size=[32] batch_size_mem=[32] model=[cnn] dataset=[cifar100] memory_size=[2000,5000,10000] lr=[0.001] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[adamw] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[cnnbn] dataset=[cifar100] memory_size=[2000,5000,10000] lr=[0.007] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[adamw] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[resnet18] dataset=[cifar100] memory_size=[2000,5000,10000] lr=[0.003] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[adamw] --yes --label base &
wait

guild run main.py batch_size=[32] batch_size_mem=[32] model=[mobilenet_pretrained] dataset=[img] memory_size=[4000,10000,20000] lr=[0.0003] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[adamw] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[resnet18_pretrained] dataset=[img] memory_size=[4000,10000,20000] lr=[0.0001] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[adamw] --yes --label base &
wait

# sgd

guild run main.py batch_size=[32] batch_size_mem=[32] model=[cnn] dataset=[cifar10] memory_size=[200,500,1000] lr=[0.1] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[sgd] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[cnnbn] dataset=[cifar10] memory_size=[200,500,1000] lr=[0.7] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[sgd] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[resnet18] dataset=[cifar10] memory_size=[200,500,1000] lr=[0.05] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[sgd] --yes --label base &
wait

guild run main.py batch_size=[32] batch_size_mem=[32] model=[cnn] dataset=[cifar100] memory_size=[2000,5000,10000] lr=[0.3] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[sgd] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[cnnbn] dataset=[cifar100] memory_size=[2000,5000,10000] lr=[0.7] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[sgd] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[resnet18] dataset=[cifar100] memory_size=[2000,5000,10000] lr=[0.1] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[sgd] --yes --label base &
wait

guild run main.py batch_size=[32] batch_size_mem=[32] model=[mobilenet_pretrained] dataset=[img] memory_size=[4000,10000,20000] lr=[0.05] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[sgd] --yes --label base &
guild run main.py batch_size=[32] batch_size_mem=[32] model=[resnet18_pretrained] dataset=[img] memory_size=[4000,10000,20000] lr=[0.03] external=[none] weight_decay=0.0 epochs=1 strategy=[er] trials=10 online=True selection_strategy=[random] steps=[one,two] mode=[old] optimizer=[sgd] --yes --label base &
wait

