import AE_train_data as dim

if __name__=="__main__":
    env = "InvertedPendulum-v2"
    seed = 17
    dat, dat_shape = dim.create_train_data(seed, env)
    print(dat)
