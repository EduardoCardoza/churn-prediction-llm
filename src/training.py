def train(
    self,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = False,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "f1",
    log_mlflow: bool = False,
) -> None:
    """
    Entrena el modelo. Si tune=True, optimiza hiperparámetros.
    Si log_mlflow=True, registra parámetros y métricas en MLflow.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Variable objetivo.
    tune : bool, optional
        Si True, ejecuta RandomizedSearchCV (default False).
    n_iter : int, optional
        Número de combinaciones a probar en el tuning (default 20).
    cv : int, optional
        Número de folds en validación cruzada (default 5).
    scoring : str, optional
        Métrica de optimización (default 'f1').
    log_mlflow : bool, optional
        Si True, registra experimento activo en MLflow (default False).
    """
    import mlflow

    ModelClass = MODEL_REGISTRY[self.model_name]

    if tune and self.model_name in PARAM_GRIDS:
        print(f"[ModelTrainer] Tuning {self.model_name} ({n_iter} iter)...")
        base_model = ModelClass(random_state=self.random_state) \
            if self.model_name != "logistic_regression" \
            else ModelClass()

        search = RandomizedSearchCV(
            base_model,
            param_distributions=PARAM_GRIDS[self.model_name],
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        print(f"[ModelTrainer] Mejor {scoring} (CV): {search.best_score_:.4f}")

        if log_mlflow:
            mlflow.log_params(self.best_params)
            mlflow.log_param("tuning_method", "RandomizedSearchCV")
            mlflow.log_param("n_iter", n_iter)
            mlflow.log_metric(f"best_cv_{scoring}", round(search.best_score_, 4))
    else:
        print(f"[ModelTrainer] Entrenando {self.model_name} sin tuning...")
        self.model = ModelClass(random_state=self.random_state) \
            if self.model_name != "logistic_regression" \
            else ModelClass(max_iter=1000)
        self.model.fit(X_train, y_train)

        if log_mlflow:
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("tune", False)

    print(f"[ModelTrainer] Entrenamiento completado ✓")