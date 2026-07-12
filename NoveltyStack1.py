def graded_z(q, proto):
    """dot(q, proto) / sqrt(sum(proto[q!=0]**2)).  Null ~ N(0,1); family z ~90-362; ceiling sqrt(D_g)=362."""
    mask = q != 0
    var = float(np.sum(proto[mask].astype(np.int64) ** 2))
    if var <= 0:
        return 0.0
    dot = int(np.dot(q.astype(np.int32), proto.astype(np.int32)))
    return dot / np.sqrt(var)
