import torch
from rich.console import Console
from rich.table import Table
from models.fusion import DQGMSD

from models import DQGMSD
from data import MultimodalSarcasmDataset

console = Console()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQGMSD().to(device)
    
    # Simulate a multimodal batch
    ids = torch.randint(0, 1000, (1, 16)).to(device)
    mask = torch.ones((1, 16)).to(device)
    video = torch.rand((1, 8, 3, 112, 112)).to(device)
    audio = torch.randn((1, 16000)).to(device)

    logits, weights = model(ids, mask, video, audio)
    
    table = Table(title="Dynamic Quality Gating Weights")
    table.add_column("Modality", style="cyan")
    table.add_column("Weight", style="magenta")
    table.add_row("Text", f"{weights[0,0]:.4f}")
    table.add_row("Video", f"{weights[0,1]:.4f}")
    table.add_row("Audio", f"{weights[0,2]:.4f}")
    
    console.print(table)
    console.print(f"[bold green]Prediction Ready![/bold green]")

if __name__ == "__main__":
    main()