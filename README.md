# F(inn)ish hash table

Hash table for insert-heavy workloads where arena allocation can be used for allocating the internal buckets. The implementation here is more of a reference for future me and contains some useless stuff and is also missing out on the arena allocator support.

```golang 
m := New[K, V]()

m.Put(k, v)
v, ok := m.Get(k)
m.Delete(k)
```

* Resizing work is done incrementally.
* [Memory use also grows incrementally](#memory). In the common case it grows by `1 * sizeof(bucket)` bytes.
* Resizing usually touches only a very small amount of memory so it doesn't end up polluting your CPU caches.
* Extremely arena allocator friendly.
    * Allocating the trie with a general purpose allocator but the buckets from an arena makes my compiler go hard. The symbol table really loves this.
* Lots of little optimization opportunities that a language more eloquent than Go could realize.
* Uhhh...

See the [benchmarks section](#benchmarks).

## Concurrent Fish table

```go
m := NewCon[K, V]()
m2 := NewCon2[K, V]() // for lock-free Get()s

m.Put(k, v)
v, ok := m.Get(k)
m.Delete(k)
```

* Resizing work is done incrementally.
* [Memory use also grows incrementally](#memory). In the common case it grows by `1 * sizeof(bucket)` bytes.
* Resizing usually touches only a very small amount of memory so it doesn't end up polluting your CPU caches.
* Extremely arena allocator friendly.
    * Allocating the trie with a general purpose allocator but the buckets from an arena makes my compiler go hard. The symbol table really loves this.
* Is broken until [golang/go/issues/19057](https://github.com/golang/go/issues/19057) is implemented.
    * A very useful optimization depends on this.
* Per-bucket mutexes for fine-grained locking. Most buckets only have population of 8 to 16 entries so contention is unlikely.
    * `ConFishTable2` also uses per-bucket version number to provide lock-free access for `Get`s in the common case.
        * But it might be buggy. Not tested. Don't use. Broken. Exists just to lie in the benchmark section.
* Goroutines that `Put` are forced to help with resizing work if there's lots to do.
    * For large amounts of resizing work multiple goroutines are spawned to ensure that multiple CPU cores help with it.
        * These spawned goroutines could actually do all of the resizing work but I prefer to have the `Put`ters help.
* Lots of little optimization opportunities that a language more eloquent than Go could realize.
* Uhhh...

See the [benchmarks section](#benchmarks).

## Consistent hashing

```go
m := NewConsistent[K, V]()

m.Put(k, v)
k, v, ok := m.GetClosest(k2)
m.Delete(k)
```

* ~O(1) `Put`, `GetClosest` & `Delete`

# Benchmarks

It seems to me that hash table benchmarks mostly just measure the number of cache-misses per operation. Concurrent hash tables on the other hand seem to measure nothing but cache-misses and lock contention. So I made sure that the numbers below are lies and do not represent any useful workload.

    goos: linux
    goarch: amd64
    cpu: Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz

## FishTable

**NOTE**: The `Lookup` benchmarks use a hash table with `ops*1` entries. The `Insert` benchmarks initialize the table to capacity of 4.

    go test --exec "taskset -c 0" --bench Benchmark...

    Lookup/ops=90         9.45 ns/lookup
    Lookup/ops=10000     11.76 ns/lookup
    Lookup/ops=1000000   53.25 ns/lookup

    Insert/ops=10000     30.19 ns/insert       302771 B/iter       931 allocs/iter
    Insert/ops=1000000   58.12 ns/insert     27865842 B/iter     89552 allocs/iter

## ConFishTable & ConFishTable2 (Concurrent)

**NOTE**: The concurrent `Lookup` benchmarks use a hash table with `ops*50` entries. The `Mixed` benchmarks pre-populate the hash table to `ops/10` entries. The `Batches` benchmarks use the batch APIs that operate on 8 keys while also taking advantage of software prefetching. The `Insert` benchmarks initialize the table to capacity of 16.

    go test --exec "taskset -c 0-7" --bench BenchmarkConParallel...

    Lookup/ops=1000             228.4 million_lookups/s
    Lookup2/ops=1000            341.1 million_lookups/s
    LookupBatches/ops=1000      270.8 million_lookups/s
    LookupBatches2/ops=1000     471.2 million_lookups/s

    Lookup/ops=100000            95.3 million_lookups/s
    Lookup2/ops=100000          146.5 million_lookups/s
    LookupBatches/ops=100000    167.9 million_lookups/s
    LookupBatches2/ops=100000   248.9 million_lookups/s

    Insert/size=100000    32.75 million_inserts/s   3125509 B/iter    8996 allocs/iter
    Insert2/size=100000   29.05 million_inserts/s   3125732 B/iter    8996 allocs/iter
    
    Mixed/ops=100000    114.2 million_ops/s
    Mixed2/ops=100000   127.9 million_ops/s
    
## Memory

### How the overall memory use grows as the map grows

```go
m := New[uint64, uint64]()
for i := range 1000000000 {
    measureSystemMemoryUsage(m.Len())
    m.Put(i, i)
}
```

Below the `heap` is `runtime/MemStats.HeapSys` aka the amount of virtual memory requested from the OS and it clearly shows the incremental memory use when growing. Compare this to your usual hash table which usually doubles (sometimes 3x) the amount of used memory. Even hash tables doing `realloc` don't behave this nicely. Also consider that the `sizeof(K+V)` here is 16 bytes and that at 1000k entries we have requested just ~31MiB from the OS to store our ~16MiB of entries.

    m.Len() heap
    0       3MiB
    10000   3MiB
    20000   3MiB
    30000   3MiB
    40000   3MiB
    50000   3MiB
    60000   3MiB
    70000   7MiB
    80000   7MiB
    90000   7MiB
    100000  7MiB
    110000  7MiB
    120000  7MiB
    130000  7MiB
    140000  7MiB
    150000  7MiB
    160000  7MiB
    170000  7MiB
    180000  7MiB
    190000  7MiB
    200000  11MiB
    210000  11MiB
    220000  11MiB
    230000  11MiB
    240000  11MiB
    250000  11MiB
    260000  11MiB
    270000  11MiB
    280000  11MiB
    290000  11MiB
    300000  11MiB
    310000  11MiB
    320000  15MiB
    330000  15MiB
    340000  15MiB
    350000  15MiB
    360000  15MiB
    370000  15MiB
    380000  15MiB
    390000  15MiB
    400000  15MiB
    410000  15MiB
    420000  15MiB
    430000  15MiB
    440000  15MiB
    450000  15MiB
    460000  15MiB
    470000  19MiB
    480000  19MiB
    490000  19MiB
    500000  19MiB
    510000  19MiB
    520000  19MiB
    530000  19MiB
    540000  19MiB
    550000  19MiB
    560000  19MiB
    570000  23MiB
    580000  23MiB
    590000  23MiB
    600000  23MiB
    610000  23MiB
    620000  23MiB
    630000  23MiB
    640000  23MiB
    650000  23MiB
    660000  23MiB
    670000  23MiB
    680000  23MiB
    690000  23MiB
    700000  23MiB
    710000  27MiB
    720000  27MiB
    730000  27MiB
    740000  27MiB
    750000  27MiB
    760000  27MiB
    770000  27MiB
    780000  27MiB
    790000  27MiB
    800000  27MiB
    810000  27MiB
    820000  27MiB
    830000  27MiB
    840000  27MiB
    850000  27MiB
    860000  27MiB
    870000  31MiB
    880000  31MiB
    890000  31MiB
    900000  31MiB
    910000  31MiB
    920000  31MiB
    930000  31MiB
    940000  31MiB
    950000  31MiB
    960000  31MiB
    970000  31MiB
    980000  31MiB
    990000  31MiB
    1000000 31MiB
    1010000 35MiB
    1020000 35MiB
    1030000 35MiB
    1040000 35MiB
    1050000 35MiB
    1060000 35MiB
    1070000 35MiB
    1080000 35MiB
    1090000 35MiB
    1100000 35MiB
    1110000 35MiB
    1120000 35MiB
    1130000 39MiB
    1140000 39MiB
    1150000 39MiB
    1160000 39MiB
    1170000 39MiB
    1180000 39MiB
    1190000 39MiB
    1200000 39MiB
    1210000 39MiB
    1220000 39MiB
    1230000 39MiB
    1240000 39MiB
    1250000 39MiB
    1260000 43MiB
    1270000 43MiB
    1280000 43MiB
    1290000 43MiB
    1300000 43MiB
    1310000 43MiB
    1320000 43MiB
    1330000 43MiB
    1340000 43MiB
    1350000 43MiB
    1360000 43MiB
    1370000 43MiB
    1380000 43MiB
    1390000 43MiB
    1400000 43MiB
