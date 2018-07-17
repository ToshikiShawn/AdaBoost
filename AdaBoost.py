import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class AdaBoost():
    def WLS(self, X_train, y_train, W):
        '''
        重み付き最小二乗法によって、弱識別器のパラメータβを推定する
        '''
        Weights = np.diag(W)
        A = np.dot(np.dot(X_train.T, Weights), X_train)
        B = np.dot(np.dot(X_train.T, Weights), y_train)
        beta = np.dot(np.linalg.inv(A), B)
        return beta

    def judge(self, X_train, y_train, beta):
        '''
        推定値と実際のyの値がどれくらい一致しているかを調べる。
        Iは、次元がデータ数のベクトルで、推定値と実際の値が一致していれば0,一致していなければ1に各要素がなっている。
        '''
        # yの予測値を0~1の連続値で計算
        y_predict = np.dot(X_train, beta)
        # yの予測値で0.5を閾値として、2値分類を行う
        y_predict = (y_predict > 0.5) * 1
        I = (y_predict != y_train) * 1
        return I

    def Boosting(self, X_train, y_train, M = 100):
        '''
        モデル学習を行う。Mはアンサンブル学習における弱学習器の個数。
        それぞれの弱学習器とその信頼度を出力する。
        '''
        beta_out = np.empty((X_train.shape[1], 0), float)
        alpha_out = []
        # 弱学習器への重みを初期化する
        W = (np.ones((X_train.shape[0]), float)) / X_train.shape[0]
        for m in range (M):
            # 重み付き最小二乗法によって線形識別器のパラメータを推定する
            beta = self.WLS(X_train, y_train, W)
            # I: y_predictとy_trainがどのくらい一致していないか、　、　
            I = self.judge(X_train, y_train, beta)
            # eps: 誤り率
            eps = np.sum(np.dot(W, I))/np.sum(W)
            # 誤り率が0になったらbreakする
            if eps == 0:
                return beta_out, np.array(alpha_out)
            # alpha: 各弱学習器の信頼度を計算する
            alpha = np.log((1 - eps)/eps)
            # 重みを更新する
            W = W * np.exp(I * alpha)
            beta_out = np.append(beta_out, beta.reshape(X_train.shape[1], 1), axis = 1)
            alpha_out.append(alpha)
        return beta_out, np.array(alpha_out)

# データセットを読み込む
breast_cancer = datasets.load_breast_cancer()
X, y =  breast_cancer.data, breast_cancer.target
# データをテストデータと訓練データに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
Ada = AdaBoost()
beta_out, alpha_out = Ada.Boosting(X_train, y_train, M = 100)
y_pred = np.dot(X_test, np.dot(beta_out, alpha_out))
y_pred = (y_pred > 0.5) * 1
matrix =  confusion_matrix(y_test, y_pred)
print(matrix) 
