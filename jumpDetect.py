import pandas as pd

MILLI = 1000

EARTH_GRAVITY = 9.81
ACCELERATION_ERROR = .5 * EARTH_GRAVITY
INTERPOLATION_SPAN = 100

MIN_N_FRAMES = 4

class JumpCounter:
    def __init__(self):
        self.timestamps = []
        self.boxes = []
        self.allTimestamps = []
        self.allBoxes = []
        self.count = 0
        self.lastJumpTimestamp = None
        self.jumpHeight = []
        self.manHeight = 1.7

    def checkForJump(self) -> bool:
        df = self.df
        m_to_p_ratio = self.manHeight / df.box.head(1).item()[3]
        
        df.index = pd.to_datetime(df.index, unit='ms')
        df['y'] = df.box.apply(lambda r: - r[1] * m_to_p_ratio)
        interpolated = df.y.resample('1L').interpolate()
        smoothed = interpolated.ewm(span=.5*INTERPOLATION_SPAN).mean()
        velocity = (smoothed.diff() * MILLI).ewm(span=INTERPOLATION_SPAN).mean()
        acceleration = (velocity.diff() * MILLI).ewm(span=INTERPOLATION_SPAN).mean()

        
        personHeight = m_to_p_ratio * df.box[-1][-1]
        self.jumpHeight.append((self.boxes[0][1] - self.boxes[-1][1]) * m_to_p_ratio)

        df = pd.DataFrame({
            'y': smoothed,
            'v': velocity,
            'a': acceleration.shift(-20)
        })
        df['freefall'] = ((df.a + EARTH_GRAVITY).abs() < ACCELERATION_ERROR)
        df['local_maximum'] = ((df.y.shift(1) < df.y) & (df.y.shift(-1) <= df.y))
        df['high_enough'] = (df.y - df.y.min()) > personHeight * 0.1

        if any(df.freefall & df.local_maximum & df.high_enough):
            self.boxes = self.boxes[-MIN_N_FRAMES:]
            self.timestamps = self.timestamps[-MIN_N_FRAMES:]
            return True

        return False

    def countJumps(self, box:tuple, timestamp:float):
        if box is None:
            return self.count
        
        self.boxes.append(box)
        self.timestamps.append(timestamp)
        self.allBoxes.append(box)
        self.allTimestamps.append(timestamp)

        if len(self.boxes) > 4 * INTERPOLATION_SPAN:
            self.boxes = self.boxes[:INTERPOLATION_SPAN]
            self.timestamps = self.timestamps[:INTERPOLATION_SPAN]

        if self.checkForJump():

            self.count += 1
            self.lastJumpTimestamp = timestamp

    
    def setPersonHeight(self, h:float):
        self.manHeight = h

    def getJumpHeight(self) -> float:
        return self.jumpHeight
    
    def getJumpCounts(self) -> int:
        return self.count
    
    @property
    def df(self):
        return pd.DataFrame({
            'box': self.boxes
        }, index=self.timestamps)

    @property
    def all_df(self):
        return pd.DataFrame({
            'box': self.allBoxes
        }, index=self.allTimestamps)
