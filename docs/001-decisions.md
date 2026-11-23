# decisions

| decision | reason |
|---|---|
| keep the [experiment numbers](../experiments/README.md) | readers can follow the work in order without giving old experiments new identities. |
| ship code and aggregate tables, not scans or fitted arrays | the generated-data example runs without IXI, and each plot can be regenerated from its released inputs. |
| distinguish projections from measurements | a sampled one-core PCNtoolkit run is not a timed full-mask run or a multi-core benchmark. |
| label the large design as synthetic | it changes both the data and reference size; the two workloads do not establish a scaling curve. |
| treat the recurrence rules separately | the at-least-52 rule and the all-64 rule came from different runs and answer different questions. |
| keep the reference package optional | PCNtoolkit is installed separately for baselines and comparisons; see [licences](licences.md). |
