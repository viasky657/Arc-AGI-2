import matplotlib.pyplot as plt
import numpy as np
import imageio

def parse_data(data):
    models = {}
    lines = data.strip().split('\n')
    current_model = None
    for line in lines:
        if 'Model:' in line:
            current_model = line.split('Model:')[1].strip()
            models[current_model] = {}
        elif ':' in line and '/' in line and current_model:
            parts = line.split(':')
            category = parts[0].strip()
            try:
                score = float(parts[1].split('/')[0].strip())
                models[current_model][category] = score
            except (ValueError, IndexError):
                pass
    return models

def create_graph(models, output_filename='evaluation_graph.gif'):
    model_names = list(models.keys())
    categories = list(next(iter(models.values())).keys())
    
    # Filter out 'Overall Score' for individual category comparison
    plot_categories = [cat for cat in categories if cat != 'Overall Score']
    
    n_models = len(model_names)
    n_categories = len(plot_categories)
    
    # Increase figure width to accommodate legend
    fig, ax = plt.subplots(figsize=(16, 8))
    
    index = np.arange(n_categories)
    bar_width = 0.15
    opacity = 0.8
    
    colors = ["#2A2B70", "#2D464E", '#B0C4DE', '#778899']
    
    for i, model_name in enumerate(model_names):
        scores = [models[model_name].get(cat, 0) for cat in plot_categories]
        ax.bar(index + i * bar_width, scores, bar_width,
               alpha=opacity,
               color=colors[i % len(colors)],
               label=model_name)

    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Model Performance Evaluation', fontsize=14)
    ax.set_xticks(index + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(plot_categories, rotation=45, ha="right")
    
    # Move legend outside the plot area to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Create a GIF
    with imageio.get_writer(output_filename, mode='I', duration=1) as writer:
        fig.canvas.draw()
        # Get ARGB buffer and convert to RGB
        buf = fig.canvas.tostring_argb()
        image = np.frombuffer(buf, dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, 1:]  # Drop alpha channel and reorder to RGB
        writer.append_data(image)

    plt.close(fig)

if __name__ == "__main__":
    data = """
    Model: Inception/Mercury
    Moral Behavior: 1.00/1.0
    Willingness To Help: 1.00/1.0
    Emotion Recognition: 0.94/1.0
    Cheering Up: 0.69/1.0
    Assistance Needed: 1.00/1.0
    Assistance Appropriateness: 0.98/1.0
    Overall Score: 0.94/1.0

    Model: Qwen/Qwen3-30b-a3b
    Moral Behavior: 1.00/1.0
    Willingness To Help: 1.00/1.0
    Emotion Recognition: 0.97/1.0
    Cheering Up: 0.80/1.0
    Assistance Needed: 0.99/1.0
    Assistance Appropriateness: 0.99/1.0
    Overall Score: 0.96/1.0

    Model: Anthropic/Sonnet 4.0
    Moral Behavior: 1.00/1.0
    Willingness To Help: 1.00/1.0
    Emotion Recognition: 0.96/1.0
    Cheering Up: 0.73/1.0
    Assistance Needed: 1.00/1.0
    Assistance Appropriateness: 1.00/1.0
    Overall Score: 0.95/1.0

    Model: Openai/04 Mini high
    Moral Behavior: 1.00/1.0
    Willingness To Help: 1.00/1.0
    Emotion Recognition: 0.94/1.0
    Cheering Up: 0.73/1.0
    Assistance Needed: 0.98/1.0
    Assistance Appropriateness: 0.98/1.0
    Overall Score: 0.94/1.0
    """
    
    models_data = parse_data(data)
    create_graph(models_data)
    print("Graph saved as evaluation_graph.gif")
