class AbstractView:
    
    def update_data(self, data):
        raise NotImplementedError
        
    def add_trajectory(self, trajectory):
        raise NotImplementedError
        
    def add_trajectories(self, trajectories):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError