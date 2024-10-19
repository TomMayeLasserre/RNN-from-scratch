import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialisation des poids
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Poids d'entrée vers état caché
        self.U = np.random.randn(input_size, hidden_size) * 0.1
        # Poids état caché vers état caché
        self.W = np.random.randn(hidden_size, hidden_size) * 0.1
        # Poids état caché vers sortie
        self.V = np.random.randn(hidden_size, output_size) * 0.1

        # Biais
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, inputs):
        T = len(inputs)
        self.hs = {}
        self.hs[-1] = np.zeros((1, self.hidden_size))
        self.outputs = []

        for t in range(T):
            x_t = inputs[t].reshape(1, -1)
            self.hs[t] = self.tanh(
                np.dot(x_t, self.U) + np.dot(self.hs[t - 1], self.W) + self.bh
            )
            # Pas d'activation en sortie
            y_t = np.dot(self.hs[t], self.V) + self.by
            self.outputs.append(y_t)

        return self.outputs

    def backward(self, inputs, target, learning_rate):
        T = len(inputs)
        # Initialisation des gradients
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # Initialisation de l'erreur à l'état caché suivant
        dh_next = np.zeros((1, self.hidden_size))

        # Calcul du gradient de la perte à l'instant final
        dy = (self.outputs[-1] - target.reshape(1, -1))  # Gradient de la perte
        dV += np.dot(self.hs[T - 1].T, dy)
        dby += dy

        dh = np.dot(dy, self.V.T)

        for t in reversed(range(T)):
            x_t = inputs[t].reshape(1, -1)
            dh_total = dh + dh_next
            dh_raw = dh_total * self.tanh_derivative(self.hs[t])

            dbh += dh_raw
            dU += np.dot(x_t.T, dh_raw)
            dW += np.dot(self.hs[t - 1].T, dh_raw)
            dh_next = np.dot(dh_raw, self.W.T)

        # Clip gradients to prevent exploding gradients
        for dparam in [dU, dW, dV, dbh, dby]:
            np.clip(dparam, -1, 1, out=dparam)

        # Mise à jour des poids et des biais
        self.U -= learning_rate * dU
        self.W -= learning_rate * dW
        self.V -= learning_rate * dV
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

    def train(self, X_train, y_train, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for inputs, target in zip(X_train, y_train):
                self.outputs = self.forward(inputs)
                loss = self.compute_loss(self.outputs[-1], target)[0][0]
                total_loss += loss
                self.backward(inputs, target, learning_rate)

            if epoch % 10 == 0:
                avg_loss = total_loss / len(X_train)
                losses.append(avg_loss)
                print(f"Epoch {epoch}, Loss: {avg_loss}")
        return losses

    def compute_loss(self, y_pred, y_true):
        loss = 0.5 * (y_true - y_pred) ** 2
        return loss

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs[-1]  # Retourne la dernière sortie
