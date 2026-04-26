class RoomClient:
    def __init__(self, name):
        self.name = name

    def send_message(self, message):
        print(f"{self.name} sends: {message}")



# create  clients
client1 = RoomClient("Alice")
client2 = RoomClient("Bob")
client3 = RoomClient("Charlie")
client4 = RoomClient("Diana")