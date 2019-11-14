# Simple Bidirection LSTM model
Genereate/predict a sentence using a Bidirectionl based LSTM model.

# Dependencies  
For this simple code, we only need pytorch, numpy, and visdom. No other libraries required!  
We use visdom to graph our loss value during training as seen in the picture below!

# How To Use
**1. Clone git**
```
$ git clone httpshttps://github.com/philgookang/simple_bilstm_model.git
$ cd simple_bilstm_model
```

**2. Train the model**  
To train the model yourself, run the following code. If you want to use a pretrained model, go direction to setp 3.  
```
$ python train.py
```

**3. Test the model**  
To test the model, change the value for line 13 and run the following code.
```
$ python test.py
```

# Loss graph
This is a picture of the loss value during training. While it may look very unclean, the dataset is only 20 sentences.  
But it is converging if you look carefully! :)
![visdom training loss graph](https://github.com/philgookang/simple_bilstm_model/blob/master/loss_graph.png?raw=true)
