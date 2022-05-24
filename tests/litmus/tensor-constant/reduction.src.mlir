// VERIFY
// ARGS: -max-const-tensor-size=1

func.func @reduction() -> tensor<96xf32> {
  %cst = arith.constant dense<[0.0682313442, 0.11823535, 0.00101596117, -0.832744777, 0.140271246, -0.004596374, -0.0096078515, -0.160056219, -15.1624937, 0.0579887033, -5.60889578, 6.32281113, 0.0182797015, -0.62245369, -4.38118076, 3.12853169, 0.0507111549, -0.0085429633, 1.29039705, 1.12123299, 0.148220479, 0.0438260138, 1.70479298, 2.11868453, -0.0807631239, 0.103669681, -4.32983637, -0.214988053, -15.7382526, 1.80917239, -0.0593759641, 0.473805666, 1.25436127, 1.40065825, -0.205554545, -0.995822668, 0.232252926, -6.12770891, 1.15569079, 1.07830524, 0.870261549, 1.48300898, -13.8531265, 0.0466267169, -0.0185232349, 3.2115159, 0.0539550185, 2.34955549, 5.27346516, 0.00543922186, -0.0657430738, 0.0674407482, 0.594510794, -0.0104416143, 0.0745761394, 0.0263969172, -0.551691234, 1.24849343, 3.63749719, 1.07425141, 1.96152425, 1.22531879, 0.0472926088, 1.21263051, 0.0238813534, 0.102453589, -4.62648582, 1.36267805, 0.0433612764, 0.583995938, 0.637062549, 0.617556154, -0.0675817207, 0.0921492204, 1.175125, -0.530795097, 0.0533812642, -0.321550846, -0.0194551907, 1.71084678, -0.100901172, 0.636657178, 1.60127342, 1.28854251, 9.08566379, 0.511179686, 1.19849312, 1.39879596, 0.225112975, 0.0105168037, -2.95897365, 1.23447859, 2.69905615, 0.625758052, 1.38598251, -0.142109096]> : tensor<96xf32>
  return %cst : tensor<96xf32>
}
