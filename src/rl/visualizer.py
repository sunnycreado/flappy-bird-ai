import neat
import pickle

class NEATVisualizer:
    def __init__(self, config_path: str):
        """Initialize the visualizer"""
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

    def load_genome(self, genome_path: str):
        """Load a trained genome"""
        with open(genome_path, 'rb') as f:
            genome = pickle.load(f)
        return genome

    def create_network(self, genome):
        """Create neural network from genome"""
        return neat.nn.FeedForwardNetwork.create(genome, self.config)

    def visualize_network(self, genome):
        """Visualize the neural network structure"""
        try:
            import visualize
            visualize.draw_net(self.config, genome, True)
        except ImportError:
            print("Visualization dependencies not installed") 