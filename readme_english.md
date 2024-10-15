# PyTorch TP

## Preliminary Steps

1. Download this project:

`git clone https://gitlab.inria.fr/chxu/pytorch_exercice.git`


2. Create an environment for this project using the pytorch.yml file.

3. Execute the program:

4. python3 main.py --experiment "faces"

## Questions:

1. Where is the data for the "faces" task, and what is the data?
2. What is the purpose of the "faces" task?
3. Which model does the "faces" task choose?
4. What loss function does the "faces" task use?
5. What algorithm does the "faces" task apply for learning?
6. Where are the definitions for the arguments of this program? Understand each argument with its default value.

## Exercise

1. Complete the accuracy function in the metric.py file, which returns the accuracy of the prediction.

2. Evaluate the model's accuracy on the test dataset in the main.py file. Re-run the program; what is your observation regarding the test accuracy compared to the training accuracy?

3. Run the program using a different batch size:

    python3 main.py --experiment "faces" --batch_size 4

    What is your observation?

4. Run the program using a different learning rate:

    python3 main.py --experiment "faces" --lr 1e-3

    What is your observation?

5. Add the choice of the Mini-batch Gradient Descent algorithm by calling "sgd."

6. Add a Fashion-MNIST dataset to the program and call the task "fash_mnist."
    Hint: You need to modify the args.py, models.py, loader.py, and trainer.py files. Run the program with the appropriate arguments.
