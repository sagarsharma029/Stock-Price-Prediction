# Stock-Price-Prediction
This project uses LSTM RNN to train on existing data about stocks (TSLA, GOOGL, TATA and others) and then predcit the prices by using the trained model.                     
model trainer.py is used to train our model based on LSTM on the existing data and save the trained model and its weights.                                            
predictor.py is used to predict future prices using our pre-trained model.                                                                                                           
Just be careful to use same stock data for training the model and predicting the prices (testing).                                                                                 
Added pre-trained model(model.json and model-weights.hdf5) for TSLA stock. Train for other stocks yourself.
