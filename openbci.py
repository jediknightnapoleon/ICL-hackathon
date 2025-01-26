#! /usr/bin/env python3

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations
import os
import csv
import numpy as np
import time

class NeuralOscillations():
    def __init__(self, sound_duration=0.5, sampling_rate=44100, timeout=0, board_id=BoardIds.SYNTHETIC_BOARD, ip_port=0,
                 ip_protocol=0, ip_address='', serial_port='', mac_address='',
                 streamer_params='', serial_number='', file='', master_board=BoardIds.NO_BOARD):
        self.sound_duration = sound_duration
        self.sampling_rate = sampling_rate
        self.timeout = timeout
        self.board_id = board_id
        self.ip_port = ip_port
        self.ip_protocol = ip_protocol
        self.ip_address = ip_address
        self.serial_port = serial_port
        self.mac_address = mac_address
        self.streamer_params = streamer_params
        self.serial_number = serial_number
        self.file = file
        self.master_board = master_board

        self.wf = 3 # BLACKMAN_HARRIS Window Function
        self.num_samples = 512

    """
    timeout: int = field(default=0)
    #board_id: int# = field(default=BoardIds.SYNTHETIC_BOARD)
    ip_port: int = field(default=0)
    ip_protocol: int = field(default=0)
    ip_address: str = field(default='')
    #serial_port: str# = field(default='')
    mac_address: str = field(default='')
    streamer_params: str = field(default='')
    serial_number: str = field(default='')
    file: str = field(default='')
    master_board: int = field(default=BoardIds.NO_BOARD)

    num_samples: int = field(default=512)
    window_function: int = field(default=3)

    def __init__(self, board_id, serial_port):
        self.board_id = board_id
        self.serial_port = serial_port
    """

    def initialise_board(self):
        params = BrainFlowInputParams()
        params.timeout = self.timeout
        params.board_id = self.board_id
        params.ip_port = self.ip_port
        params.ip_protocol = self.ip_protocol
        params.ip_address = self.ip_address
        params.serial_port = self.serial_port
        params.mac_address = self.mac_address
        params.streamer_params = self.streamer_params
        params.serial_number = self.serial_number
        params.file = self.file
        params.master_board = self.master_board
        return BoardShim(self.board_id, params)


    def filter(self, eeg_data, lower_bound, upper_bound, window_function):
        psd = DataFilter.get_psd(eeg_data, BoardShim.get_sampling_rate(self.board_id), window_function)
        return DataFilter.get_band_power(psd, lower_bound, upper_bound)

    # Number of Samples has to be 512 to be compatible with all 5 waves
    def eeg_recorder(self, eeg_channel_count=8, delta=True, theta=True, alpha=True, beta=True, gamma=True):
        board = self.initialise_board()
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        board.prepare_session()
        board.start_stream()
        time.sleep(3)
        theta_pow = list()
        alpha_pow = list()
        beta_pow = list()
        gamma_pow = list()
        ta_ratios = list()
        tb_ratios = list()
        tg_ratios = list()
        ab_ratios = list()
        ag_ratios = list()
        bg_ratios = list()

        try:
            while True:
                data = board.get_current_board_data(self.num_samples)
                tMean, aMean, bMean, gMean = 0, 0, 0, 0
                for c in range(eeg_channel_count):
                    tMean += np.mean(self.filter(data[eeg_channels[c]], 4, 8, self.wf))
                    aMean += np.mean(self.filter(data[eeg_channels[c]], 8, 13, self.wf))
                    bMean += np.mean(self.filter(data[eeg_channels[c]], 13, 32, self.wf))
                    gMean += np.mean(self.filter(data[eeg_channels[c]], 32, 100, self.wf))
                tMean = tMean / eeg_channel_count
                aMean = aMean / eeg_channel_count
                bMean = bMean / eeg_channel_count
                gMean = gMean / eeg_channel_count

                # Possible Ratios (most useful rn TAR and TBR)
                tar = tMean / aMean
                tbr = tMean / bMean
                tgr = tMean / gMean

                abr = aMean / bMean
                agr = aMean / gMean

                bgr = bMean / gMean

                print(f'Theta: {tMean:.6f}\tAlpha: {aMean:.6f}\tTAR: {tar:.6f}')
                time.sleep(1)

                theta_pow.append(tMean)
                alpha_pow.append(aMean)
                beta_pow.append(bMean)
                gamma_pow.append(gMean)
                ta_ratios.append(tar)
                tb_ratios.append(tbr)
                tg_ratios.append(tgr)
                ab_ratios.append(abr)
                ag_ratios.append(agr)
                bg_ratios.append(bgr)

        except KeyboardInterrupt:
            self.create_csv(theta_pow, alpha_pow, beta_pow, gamma_pow, ta_ratios, tb_ratios, tg_ratios, ab_ratios, ag_ratios, bg_ratios)
            board.stop_stream()
            board.release_session()
            raise Exception

    def create_csv(self, *data):
        try:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"output_{timestamp}.csv"
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archive')
            file_path = os.path.join(file_path, filename)
            print(file_path)
            rows = zip(*data)
            rows = list(rows)
            print(rows)
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ["Theta_pow", "Alpha_pow", "Beta_pow", "Gamma_pow", "TAR", "TBR", "TGR", "ABR", "AGR", "BGR"]
                writer.writerow(header)
                writer.writerows(rows)
            print(f"Data has been written to {file_path}")
        except Exception as e:
            print(f"Error: {e}")



# Board Details
BOARD_ID = 0 # CYTON_BOARD
EEG_CHANNEL_COUNT = 8
# Port Details
SERIAL_PORT = "/dev/cu.usbserial-D200QZLM"
SERIAL_PORT = "/dev/cu.usbserial-DM03GRD1"

if __name__ == "__main__":
    neural_oscillations = NeuralOscillations(board_id=BOARD_ID, serial_port=SERIAL_PORT)
    # neural_oscillations = NeuralOscillations()
    neural_oscillations.eeg_recorder(eeg_channel_count=EEG_CHANNEL_COUNT)
