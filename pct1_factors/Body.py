import pandas as pd
import numpy as np
from DatabaseConn import SessionInfluxDB, CurrentTime

def Df2Body(df,measurement_name):
    _b = []
    for k, v in df.to_dict(orient="index").items():
        a = int(k[1][0:4])
        b = int(k[1][5:7])
        c = int(k[1][8:10])
        d = int(k[1][11:13])
        e = int(k[1][14:16])
        f = int(k[1][17:19])

        for i in v:
            if type(v[i]) is int:
                if np.isnan(v[i]):
                    v[i] = None
                else:
                    v[i] = float(v[i])
            elif type(v[i]) is float:
                if np.isnan(v[i]):
                    v[i] = None
                elif np.isinf(v[i]):
                    v[i] = None
            elif v[i] is not None:
                v[i] = str(v[i])

                
                

        _d = {
            "measurement": measurement_name,
            "time": CurrentTime(a, b, c, d, e, f),
            "tags": {"code": k[0]},
            "fields": v,
        }
        _b.append(_d)
    _s = SessionInfluxDB()
    _s.Write(_b)
    return _b