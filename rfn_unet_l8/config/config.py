class Config:
    def __init__(self) -> None:
        # ->> 1. Hyperparameter test
        # ->> theta_beta = 0.03125
        # self.theta_alpha = 120.0 # 4
        # self.theta_alpha = 80.0 # 1
        # self.theta_alpha = 40.0 # 2
        # self.theta_alpha = 10.0 # 3

        # ->> theta_alpha = 80.0
        # self.theta_beta = 0.03125 # 1
        # self.theta_beta = 0.0625 # 2
        # self.theta_beta = 0.125 # 3
        # self.theta_beta = 0.25 # 4

        # ->> Constant
        # self.theta_gamma = 3.0

        # ->> 2. Exploration of multi-spectral features
        # self.n_bands = 7 
        # self.img_channel_list, self.vis_channel_list = [4, 3, 2], None # RGB
        # self.save_path = "../../result/l8/rgb/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma) # RGB
        # self.img_channel_list, self.vis_channel_list = list(range(self.n_bands)), [4, 3, 2] # Seven-band
        # self.save_path = "../../result/l8/fullband/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma)

        # # ->> 3. Ablation study with respect to the bilateral message-passing step
        # # theta_alpha, theta_beta, theta_gamma = 80, .03125, 3
        # self.theta_alpha, self.theta_beta, self.theta_gamma = 80.0, 0.03125, 3.0
        # self.n_bands = 7 
        # self.img_channel_list, self.vis_channel_list = [4, 3, 2], None # RGB
        # self.save_path = "../../result/l8/wobilateral/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma)

        # ->> 4. Ablation study w.r.t. the CRF layer. Required by r2 of IEEE J-STARS.
        self.theta_alpha, self.theta_beta, self.theta_gamma = 80.0, 0.03125, 3.0
        self.n_bands = 7 
        self.img_channel_list, self.vis_channel_list = [4, 3, 2], None # RGB
        self.data_path = "../../data/l8/test/"
        self.model_path = "backbone/"
        self.save_path = "../../result/l8/wocrf/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma)
        self.save_info_fname = "rfn.csv"

        # ->> Model parameters
        self.n_classes = 4
        self.crop_height = 512
        self.crop_width = 512
        self.bilateral_compat = 10.0
        self.spatial_compat = 3.0
        self.n_iterations = 10
