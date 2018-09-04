# AdaBoost

*AdaBoost via Weighted Least Squares*

This is a code for AdaBoost via Weighted Least Squares. 

_experiment_
I used breast cancer datasets from sk-learn.
choose 80% of data as training data and 20% of data as test data.
confusion matrix of test data is as follows.
It showed accuracy of 95.6%.

_confusion matrix_

|                    |     　　　Predic class      |　　
|                    | Positive    | Negative     |  
| Actual| Positive   |      43     |    4         |
| class | Negative   |      1      |    66        |


_Algorithm_


Initialize weights
for n = 1, ..., N


<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}N\\&space;\left\{&space;w^{\left(&space;1\right)&space;}_{n}=\dfrac&space;{1}{N}\right\}&space;_{n=1}\end{aligned}"/>

for m = 1, ..., M


<img src="https://latex.codecogs.com/gif.latex?\arg&space;\min&space;_{\beta&space;}\sum&space;^{N}_{n=1}w^{\left(&space;m\right)&space;}_{n}\left(&space;y_{n}-\beta&space;_{n}\cdot&space;x_{n}\right)&space;^{2}"/>
<img src="https://latex.codecogs.com/gif.latex?h_{m}\left(&space;x_{n}\right)&space;=\begin{cases}1\left(&space;h_{m}\left(&space;x_{n}\right)&space;>0.5\right)\\&space;0\left(&space;h_{m}\left(&space;x_{n}\right)&space;\leq&space;0.5\right)&space;\end{cases}"/>
<img src="https://latex.codecogs.com/gif.latex?I(&space;h_{m}\left(&space;x_{n}\right)&space;\neq&space;y_{n})=\begin{cases}1\;&space;\;&space;\;&space;\;&space;if\;&space;\;&space;h_{m}\left(&space;x_{n}\right)&space;\neq&space;y_{n}\\&space;0\;&space;\;&space;\;&space;\;&space;otherwise\end{cases}"/>


calculate error rate and reliablity

<img src="https://latex.codecogs.com/gif.latex?\varepsilon&space;_{n}=\dfrac&space;{\sum&space;^{N}_{n=1}w^{\left(&space;m\right)&space;}_{n}I\left(&space;h_{m}\left(&space;x\right)&space;\neq&space;y_n{}\right)&space;}{\sum&space;^{N}_{n=1}w^{\left(&space;m\right)&space;}_{n}}"/>
<img src="https://latex.codecogs.com/gif.latex?\alpha&space;_{m}=\ln&space;(&space;\dfrac&space;{1-\varepsilon&space;_{m}}{\varepsilon&space;_{m}})"/>




update weights


<img src="https://latex.codecogs.com/gif.latex?w^{(m&plus;1)}_{n}=w^{(m)}_{n}\exp&space;\{&space;\alpha&space;_{m}I\left(&space;h_{m}\left(&space;x_{n}\right)\neq&space;y_{n}\right)\right\}"/>


make a predictor


<img src="https://latex.codecogs.com/gif.latex?H\left(&space;x\right)&space;=sign\left(&space;\sum&space;^{n}_{m=1}\alpha&space;_{m}h_{m}\left(&space;x_{n}\right)\right)"/>
