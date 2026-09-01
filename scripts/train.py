def main():
    args = parse_args()

    import mlflow
    import dagshub
    from dotenv import load_dotenv
    load_dotenv()

    print(f"\n{'='*50}")
    print("ENTRENAMIENTO DE MODELO (con MLflow)")
    print(f"{'='*50}")
    print(f"Modelo:  {args.model}")
    print(f"Tuning:  {args.tune}")
    print(f"Output:  {args.output}")
    print(f"{'='*50}\n")

    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USER"),
        repo_name="churn-prediction-llm",
        mlflow=True
    )
    mlflow.set_experiment("churn-prediction-telco-scripts")

    train_df = pd.read_parquet(args.train_data)
    test_df  = pd.read_parquet(args.test_data)

    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_test  = test_df.drop('Churn', axis=1)
    y_test  = test_df['Churn']

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    run_name = f"{args.model}{'_tuned' if args.tune else '_baseline'}_script"

    with mlflow.start_run(run_name=run_name):
        trainer = ModelTrainer(
            model_name=args.model,
            random_state=args.random_state
        )
        trainer.train(
            X_train, y_train,
            tune=args.tune,
            n_iter=args.n_iter,
            cv=args.cv,
            log_mlflow=True,
        )
        trainer.save(args.output)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            trainer.model, X_test, y_test,
            model_name=run_name
        )

        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "Modelo"})
        mlflow.sklearn.log_model(trainer.model, "model")

        evaluator.save_metrics([metrics], args.metrics_output)
        mlflow.log_artifact(args.metrics_output)

        evaluator.plot_confusion_matrix(
            trainer.model, X_test, y_test,
            model_name=args.model,
            save_path="artifacts/confusion_matrix_script.png"
        )
        mlflow.log_artifact("artifacts/confusion_matrix_script.png")

        print(f"\n✓ Entrenamiento y registro en MLflow completados\n")