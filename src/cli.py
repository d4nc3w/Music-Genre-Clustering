import typer
from src.pycaret.train import train_model

app = typer.Typer()

@app.command()
def train(
    name: str = typer.Option(..., help="Name of the output model"),
    clusters: int = typer.Option(3, help="Number of clusters")
):
    """
    Train PyCaret KMeans model
    """
    typer.echo(f"🚀 Starting training for model: {name}")
    
    try:
        scoring, predictions = train_model(name, clusters)
        
        typer.echo("\nTraining finished!")
        
        typer.echo("\nScoring Metrics:")
        typer.echo(scoring.to_string(index=False))
        
        typer.echo("\nSample Predictions (First 5 rows):")
        typer.echo(predictions.head().to_string(index=False))
        
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)

if __name__ == "__main__":
    app()