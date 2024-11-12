from detecthealth import detect

class Player:
    def __init__(self):
        self.hp = 100
        self.alive = True
    
    def hpCheck(self):
        updated_hp = detect()
        if updated_hp == -1:
            self.alive = True
            self.hp = 100
        elif updated_hp == -2:
            self.alive = False
        elif updated_hp < self.hp and self.alive:
            self.hp = updated_hp
            return True
        
        return False
    
user = Player()

   
        