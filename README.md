
"# LSTM-for-Sentence-Classification-in-PyTorch-Based-on-Pixnet" 


1.You need to download data from https://github.com/pixnet/2017-pixnet-hackathon-TaskOrientedBot/blob/master/opendata.md

#### 2.This model come from : https://github.com/yuchenlin/lstm_sentence_classifier , I just use Pixnet data to replace this author's data

3.This model use :
constellation.json （星座運勢）
food.json （美味食記）
makeup.json （美妝 styleMe）
medic.json （醫療保健）
mombaby.json （親子育兒）
movie.json （電影評論）
sport.json （運動體育）
travel_foreign.json （國外旅遊）
travel_taiwan.json （國內旅遊）

4.Run LSTM_sentence_classifier.py or LSTM_sentence_classifier_cuda.py to train the model(whethere you need cuda or not)

5.Class_number show how is the data classified,and I merge travel_foreign & travel_taiwan to travel. 

6.Use other pixnet article test,this model has a correct rate of 75-82%,but in other forums like PTT, the accuracy will be lower.

7.This model is for article classification,it is not doing well in too small sentence classification.
