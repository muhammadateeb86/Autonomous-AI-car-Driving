#model with 3's pickle file
import time
import msgParser
import carState
import carControl
import csv
import os
import torch
import torch.nn as nn
import numpy as np
import joblib

class TORCS_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64]):
        super(TORCS_MLP, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Dropout(0.2)
        )
        self.gear_head = nn.Linear(hidden_sizes[2], 7)  # 7 classes for gear
        self.brake_head = nn.Linear(hidden_sizes[2], 2)  # 2 classes for brake
        self.accel_head = nn.Linear(hidden_sizes[2], 1)  # Regression for accel
        self.steer_head = nn.Linear(hidden_sizes[2], 1)  # Regression for steer
    
    def forward(self, x):
        features = self.backbone(x)
        gear = self.gear_head(features)
        brake = self.brake_head(features)
        accel = torch.sigmoid(self.accel_head(features))  # [0, 1]
        steer = torch.tanh(self.steer_head(features))  # [-1, 1]
        return gear, brake, accel, steer

class Driver(object):
    def __init__(self, stage):
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage

        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.max_speed = 250

        # Define input columns (71 features, assuming focus_4 is missing)
        self.input_cols = [
            'Angle', 'SpeedX', 'SpeedY', 'SpeedZ', 'TrackPos', 'Rpm', 'Z',
            *['track_' + str(i) for i in range(19)],
            *['opponents_' + str(i) for i in range(36)],
            *['wheelSpinVel_' + str(i) for i in range(4)],
            *['focus_' + str(i) for i in range(5)]
        ]
        print(f"Input columns defined: {self.input_cols} ({len(self.input_cols)} features)")

        try:
            self.model = TORCS_MLP(input_size=len(self.input_cols))
            self.model.load_state_dict(torch.load('torcs_mlp.pth', map_location='cpu'))
            self.model.eval()
            print("Model loaded successfully (CPU only)")

            self.scaler_X = joblib.load('scaler_X.pkl')
            self.scaler_accel = joblib.load('scaler_accel.pkl')
            self.scaler_steer = joblib.load('scaler_steer.pkl')
            print("Scalers loaded successfully")

        except Exception as e:
            print(f"Error loading model or scalers: {e}")
            raise

        self.log_file_name = "sensor_data_inference.csv"
        file_exists = os.path.isfile(self.log_file_name)
        self.log_file = open(self.log_file_name, "a", newline='', encoding="utf-8")
        self.csv_writer = csv.writer(self.log_file)

        if not file_exists:
            header = [
                'timestamp',
                'angle', 'speedX', 'speedY', 'speedZ', 'trackPos', 'rpm', 'z',
                *['track_' + str(i) for i in range(19)],
                *['opponents_' + str(i) for i in range(36)],
                *['wheelSpinVel_' + str(i) for i in range(4)],
                *['focus_' + str(i) for i in range(5)],
                'accel', 'brake', 'gear', 'steer'
            ]
            self.csv_writer.writerow(header)
            self.log_file.flush()

    def init(self):
        angles = [0 for _ in range(19)]
        for i in range(5):
            angles[i] = -90 + i * 15
            angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            angles[i] = -20 + (i - 5) * 5
            angles[18 - i] = 20 - (i - 5) * 5
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)

        sensor_data = []
        for col in self.input_cols:
            if col in ['Angle', 'SpeedX', 'SpeedY', 'SpeedZ', 'TrackPos', 'Rpm', 'Z']:
                value = getattr(self.state, f'get{col}')() or 0
                sensor_data.append(value)
            elif col.startswith('track_'):
                idx = int(col.split('_')[1])
                track = self.state.getTrack() or [0] * 19
                sensor_data.append(track[idx] if idx < len(track) else 0)
            elif col.startswith('opponents_'):
                idx = int(col.split('_')[1])
                opponents = self.state.getOpponents() or [0] * 36
                sensor_data.append(opponents[idx] if idx < len(opponents) else 0)
            elif col.startswith('wheelSpinVel_'):
                idx = int(col.split('_')[1])
                wheelSpinVel = self.state.getWheelSpinVel() or [0] * 4
                sensor_data.append(wheelSpinVel[idx] if idx < len(wheelSpinVel) else 0)
            elif col.startswith('focus_'):
                idx = int(col.split('_')[1])
                focus = self.state.getFocus() or [0] * 5
                sensor_data.append(focus[idx] if idx < len(focus) else 0)

        sensor_data = np.array([sensor_data], dtype=np.float32)

        try:
            gear, brake, accel, steer = self.predict(sensor_data)
            self.control.setGear(gear[0])
            self.control.setBrake(float(brake[0]))
            self.control.setAccel(float(accel[0]))
            self.control.setSteer(float(steer[0]))
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.control.setGear(1)
            self.control.setBrake(0.0)
            self.control.setAccel(0.0)
            self.control.setSteer(0.0)

        self.log_data(self.state, self.control)
        return self.control.toMsg()

    def predict(self, sensor_data):
        gear_mapping = {0: -1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

        sensor_data_scaled = self.scaler_X.transform(sensor_data)
        sensor_data_tensor = torch.FloatTensor(sensor_data_scaled)

        with torch.no_grad():
            gear_pred, brake_pred, accel_pred, steer_pred = self.model(sensor_data_tensor)
            gear = torch.argmax(gear_pred, dim=1).numpy()
            gear = np.array([gear_mapping[g] for g in gear])
            brake = torch.argmax(brake_pred, dim=1).numpy()
            accel = self.scaler_accel.inverse_transform(accel_pred.numpy())
            steer = self.scaler_steer.inverse_transform(steer_pred.numpy())

        return gear, brake, accel, steer

    def log_data(self, state, control):
        track = state.getTrack() or [0] * 19
        opponents = state.getOpponents() or [0] * 36
        wheelSpinVel = state.getWheelSpinVel() or [0] * 4
        focus = state.getFocus() or [0] * 5

        row = [
            time.time(),
            state.getAngle() or 0,
            state.getSpeedX() or 0,
            state.getSpeedY() or 0,
            state.getSpeedZ() or 0,
            state.getTrackPos() or 0,
            state.getRpm() or 0,
            state.getZ() or 0,
        ]
        row.extend(track)
        row.extend(opponents)
        row.extend(wheelSpinVel)
        row.extend(focus)
        row.extend([
            control.getAccel(),
            control.getBrake(),
            control.getGear() or 1,
            control.getSteer()
        ])

        self.csv_writer.writerow(row)
        self.log_file.flush()

    def onShutDown(self):
        self.log_file.close()
        print("Driver shutdown")

    def onRestart(self):
        pass
