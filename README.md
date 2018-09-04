# AdaBoost

*AdaBoost via Weighted Least Squares*

This is a code for AdaBoost via Weighted Least Squares. 

_Algorithm_

'''math
Initialize weights
for n = 1, ..., N
<img src="https://latex.codecogs.com/gif.latex?\\{w^{\left(&space;1\right)&space;}_{n}=\dfrac&space;{1}{N}\\}^{N}_{n=1}"/>

for m = 1, ..., M

\arg \min _{\beta }\sum ^{N}_{n=1}w^{\left( m\right) }_{n}\left( y_{n}-\beta _{n}\cdot x_{n}\right) ^{2}
h_{m}\left( x_{n}\right) =\begin{cases}1\left( h_{m}\left( x_{n}\right) >0.5\right)\\ O\left( h_{m}\left( x_{n}\right) \leq 0.5\right) \end{cases}
I( h_{m}\left( x_{n}\right) \neq y_{n}\right)=\begin{cases}1ifh_{m}\left( x_{n}\right) \neq y_{n}\\ 0otherwise\end{cases}

calculate error rate and reliablity
\varepsilon _{m}=\dfrac {\sum ^{N}_{n=1}w^{m}_{n}I(h_{m}\left( x\right) \neq y_{n}\right)}{\sum ^{N}_{n=1}w^{m}_{n}}
\alpha _{m}=\ln ( \dfrac {1-\varepsilon _{m}}{\varepsilon _{m}\right)}

update weights
w^{m+1}_{n}=w^{m}_{n}\exp \{ \alpha _{m}I\left( h_{m}\left( x_{n}\right)neq y_{n}\right)\right\}

make a predictor
H\left( x\right) =sign\left( \sum ^{n}_{m=1}\alpha _{m}h_{m}\left( x_{n}\right))
'''
