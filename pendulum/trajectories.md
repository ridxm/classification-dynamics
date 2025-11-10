Keys in the file:
  ['X', 'y', 'ids']

X:
  Shape: (1000, 502, 2)
  Dtype: float32
  Content: Trajectories (sequences of theta, velocity)
  First trajectory shape: (502, 2)
  First trajectory first 5 time steps:
    [[-1.98159  2.63682]
 [-1.95549  2.55658]
 [-1.93019  2.47655]
 [-1.90569  2.39691]
 [-1.88199  2.3178 ]]
  First trajectory last 5 time steps:
    [[-2.09404     0.00513035]
 [-2.09399     0.00495886]
 [-2.09394     0.00478703]
 [-2.0939      0.00461503]
 [-2.09385     0.00444303]]

y:
  Shape: (1000,)
  Dtype: int32
  Content: Outcomes (0=failure, 1=success)
  First 20 outcomes: [0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 0 0 0 0 0]
  Success count: 387
  Failure count: 613
  Success rate: 38.70%

ids:
  Shape: (1000,)
  Dtype: <U10
  Content: Sequence IDs
  First 10 IDs: ['35264' '36954' '33199' '28327' '11953' '41218' '14536' '2269' '42864'
 '45488']
  Last 10 IDs: ['47956' '31706' '38254' '31923' '16753' '19067' '25455' '835' '1092'
 '30937']