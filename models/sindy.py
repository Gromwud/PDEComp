import pysindy as ps
import numpy as np


class SINDyModel:
    def __init__(self):
        differentiation_method = ps.FiniteDifference(order=2)
        feature_library = ps.PolynomialLibrary(degree=3)
        optimizer = ps.STLSQ(threshold=0.2)
        self.model = ps.SINDy(
            differentiation_method=differentiation_method,
            feature_library=feature_library,
            optimizer=optimizer,
        )

    def fit(self, u, t):
        self.model.fit(u, t=t)

    def print(self):
        self.model.print()


class PDEFIND:
    def __init__(self, u, x, t):
        self.u = u.reshape(len(x), len(t), 1)
        # self.u = u.reshape(len(t), 1)

        library_functions = [lambda x: x, lambda x: x * x]
        library_function_names = [lambda x: x, lambda x: x + x]
        feature_library = ps.PDELibrary(library_functions=library_functions,
                                function_names=library_function_names,
                                derivative_order=3, spatial_grid=x,
                                include_bias=True, is_uniform=True).fit([self.u])

        print(feature_library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)

        self.model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
        self.x = x
        self.t = t

    def fit(self):
        self.model.fit(self.u, t=self.t)

    def print(self):
        self.model.print()

    def simulate(self):
        pred = self.model.simulate(0, self.t)
        return pred

    def differentiate(self):
        u_dot_train = self.model.differentiate(self.u, self.t)
        return u_dot_train

    def predict(self):
        u_dot_train_pred = self.model.predict(self.u)
        return u_dot_train_pred

    def score(self, x_test):
        return self.model.score(x_test)

