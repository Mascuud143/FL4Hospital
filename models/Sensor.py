class Sensor:
    def __init__(self, sensor_id, MAC_address, location, room_id,name):
        self.sensor_id = sensor_id
        self.MAC_address = MAC_address
        self.location = location
        self.room_id = room_id
        self.name = name
        self.data=[]
    
    def get_sensor_id(self):
        return self.sensor_id
    def get_MAC_address(self):
        return self.MAC_address
    
    # connect
    