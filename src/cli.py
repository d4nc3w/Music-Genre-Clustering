import typer
import optuna
from optuna.study import StudyDirection
from src.pycaret.train import train_model
from src.optuna.find_hyperparemeter import objective

app = typer.Typer(help="Music Genre Clustering Manager")

@app.command()
def train(
    name: str = typer.Option(..., help="Name of the output model"),
    clusters: int = typer.Option(3, help="Number of clusters")
):
    """
    Train PyCaret KMeans model
    """
    typer.echo(f"Starting training for model: {name}")
    
    try:
        scoring, predictions = train_model(name, clusters)
        
        typer.echo("\nTraining finished!")
        
        typer.echo("\nScoring Metrics:")
        typer.echo(scoring.to_string(index=False))
        
        typer.echo("\nSample Predictions (First 5 rows):")
        typer.echo(predictions.head().to_string(index=False))
        
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)

@app.command()
def optimize(
    trials: int = typer.Option(20, help="Number of trials to run")
):
    """
    Run Optuna to find best hyperparameters.
    """
    typer.echo(f"Starting Hyperparameter Tuning (Optuna)...")
    
    storage_url = "sqlite:///music_clustering.db"
    
    try:
        study = optuna.create_study(
            direction=StudyDirection.MAXIMIZE, 
            storage=storage_url,
            study_name="kmeans_optimization",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=trials)
        
        typer.echo("\nOptimization Finished!")
        typer.secho(f"Best Silhouette Score: {study.best_value:.4f}", fg=typer.colors.GREEN, bold=True)
        
        typer.echo("Best Hyperparameters:")
        for key, value in study.best_params.items():
            typer.secho(f"  - {key}: {value}", fg=typer.colors.CYAN)
            
    except Exception as e:
        typer.secho(f"Error during optimization: {e}", fg=typer.colors.RED)

if __name__ == "__main__":
    app()