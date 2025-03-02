import simpy

class ProcessorSharing(simpy.Resource):
    def __init__(self, env, capacity=1):
        super().__init__(env, capacity)

    def request(self, *args, **kwargs):
        # Insert the new request at the end of the queue
        request = super().request(*args, **kwargs)
        if(request in self.queue):
            self.queue.remove(request)  # Remove from default position
            self.queue.append(request)  # Add to the end
        return request

    def release(self, *args, **kwargs):
        return super().release(*args, **kwargs)