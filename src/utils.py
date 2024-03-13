def check_is_ipython():
    try:
        # examine if the module is running in an IPython environment
        __IPYTHON__
        return True
    except NameError:
        return False
