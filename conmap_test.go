package fishtable

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand/v2"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestConInsert10(t *testing.T) {
	m := NewCon[uint64, uint64]()
	const c = 10 * 1000 * 100
	for k := uint64(0); k < c; k++ {
		m.Put(k, k)
	}

	seen := make(map[uint64]uint64, m.Len())
	m.All(func(k, v uint64) bool {
		if k != v {
			t.Fatalf("want %v got %v", k, v)
		}
		old, ok := seen[k]
		if ok {
			t.Fatalf("already seen %v %v %v", ok, k, old)
		}
		seen[k] = v

		// Also test Get()
		{
			v, ok := m.Get(k)
			if !ok || v != k {
				t.Fatalf("%v %v %v", ok, k, v)
			}
		}
		return true
	})
	if m.Len() != c {
		t.Fatalf("want %v got %v", c, len(seen))
	}
	if len(seen) != c {
		t.Fatalf("want %v got %v", c, len(seen))
	}
}

func TestConInsertParallel(t *testing.T) {
	m := NewCon[uint64, uint64]()
	numgoros := runtime.GOMAXPROCS(-1) * 1
	c := numgoros * 1 * 100 * 1000
	var wg sync.WaitGroup
	var failed atomic.Bool
	if false {
		p := c / numgoros
		var off int
		for range numgoros {
			wg.Add(1)
			offl := off
			off += p
			go func() {
				defer wg.Done()
				for k := uint64(offl); k < uint64(offl+p); k++ {
					m.Put(k, k)
					v, ok := m.Get(k)
					if !ok || v != k {
						failed.Store(true)
						t.Fatalf("(%v %v) %v %v %v", m.Len(), k-uint64(offl), ok, k, v)
					}
					if k%64 == 0 && failed.Load() {
						t.FailNow()
					}
				}
			}()
		}
	} else {
		for range numgoros {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for k := uint64(0); k < uint64(c); k++ {
					m.Put(k, k)
					v, ok := m.Get(k)
					if !ok || v != k {
						failed.Store(true)
						t.Fatalf("(%v %v) %v %v %v", m.Len(), k, ok, k, v)
					}
					if k%64 == 0 && failed.Load() {
						t.FailNow()
					}
				}
			}()
		}
	}

	wg.Wait()
	if t.Failed() {
		t.Fatal()
	}

	seen := make(map[uint64]uint64, m.Len())
	m.All(func(k, v uint64) bool {
		if k != v {
			t.Errorf("want %v got %v", k, v)
		}
		old, ok := seen[k]
		if ok {
			t.Errorf("already seen %v %v %v", ok, k, old)
		}
		seen[k] = v

		// Also test Get()
		{
			v, ok := m.Get(k)
			if !ok || v != k {
				t.Errorf("%v %v %v", ok, k, v)
			}
		}
		return true
	})
	if m.Len() != c {
		t.Fatalf("want %v got %v", c, m.Len())
	}
	if len(seen) != c {
		t.Fatalf("want %v got %v", c, len(seen))
	}
}

func TestConInsertParallel2(t *testing.T) {
	m := NewCon2[uint64, uint64]()
	numgoros := runtime.GOMAXPROCS(-1) * 1
	c := numgoros * 1 * 100 * 1000
	var wg sync.WaitGroup
	var failed atomic.Bool
	if false {
		p := c / numgoros
		var off int
		for range numgoros {
			wg.Add(1)
			offl := off
			off += p
			go func() {
				defer wg.Done()
				for k := uint64(offl); k < uint64(offl+p); k++ {
					m.Put(k, k)
					v, ok := m.Get(k)
					if !ok || v != k {
						failed.Store(true)
						t.Fatalf("(%v %v) %v %v %v", m.Len(), k-uint64(offl), ok, k, v)
					}
					if k%64 == 0 && failed.Load() {
						t.FailNow()
					}
				}
			}()
		}
	} else {
		for range numgoros {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for k := uint64(0); k < uint64(c); k++ {
					m.Put(k, k)
					v, ok := m.Get(k)
					if !ok || v != k {
						failed.Store(true)
						t.Fatalf("(%v %v) %v %v %v", m.Len(), k, ok, k, v)
					}
					if k%64 == 0 && failed.Load() {
						t.FailNow()
					}
				}
			}()
		}
	}

	wg.Wait()
	if t.Failed() {
		t.Fatal()
	}

	seen := make(map[uint64]uint64, m.Len())
	m.All(func(k, v uint64) bool {
		if k != v {
			t.Errorf("want %v got %v", k, v)
		}
		old, ok := seen[k]
		if ok {
			t.Errorf("already seen %v %v %v", ok, k, old)
		}
		seen[k] = v

		// Also test Get()
		{
			v, ok := m.Get(k)
			if !ok || v != k {
				t.Errorf("%v %v %v", ok, k, v)
			}
		}
		return true
	})
	if m.Len() != c {
		t.Fatalf("want %v got %v", c, m.Len())
	}
	if len(seen) != c {
		t.Fatalf("want %v got %v", c, len(seen))
	}
}

func TestConInsertDeleteLoop(t *testing.T) {
	// This test tries to show that the performance of our overflow linked-lists doesn't
	// degrade as entries are removed and then inserted again. This test is ass.
	size := 400 * 1000
	sizeg := uint64(size)
	const repeats = 1024 * 2

	m := NewCon[uint64, uint64]()

	once := func() time.Duration {
		s := time.Now()
		for k := range sizeg {
			m.Put(k, k)
		}
		for k := range sizeg {
			m.Delete(k)
		}
		if m.Len() != 0 {
			panic(m.Len())
		}
		return time.Since(s)
	}
	first := once()
	for i := range repeats - 1 {
		d := once()
		if d >= first*2 {
			t.Fatalf("first run took %v but later run %v took %v!", first, i, d)
		}
	}
}

func BenchmarkConLookups(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := NewCon[uint64, uint64]()
			for k := uint64(0); k < size; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for k := uint64(0); k < size; k++ {
					v, _ := m.Get(k)
					if v != k {
						panic("bad")
					}
				}
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/lookup")
		})
	}
}

func BenchmarkConInsert(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				m := NewCon[uint64, uint64]()
				start := time.Now()
				for k := uint64(0); k < size; k++ {
					m.Put(k, k)
				}
				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelInsert(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 1
			sizeg := size / uint64(numGoros)

			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				m := NewCon[uint64, uint64]()
				// m.Reserve(int(size) / 4)
				start := time.Now()

				var wg sync.WaitGroup
				for i := range numGoros {
					wg.Add(1)
					goff := sizeg * uint64(i)
					go func() {
						defer wg.Done()
						var rng rand.PCG
						rng.Seed(rand.Uint64(), rand.Uint64())

						for range sizeg {
							k := goff + rng.Uint64()
							m.Put(k, k)
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*uint64(b.N))*float64(1000*1000)), "million_inserts/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelInsert2(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 1
			sizeg := size / uint64(numGoros)

			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				m := NewCon2[uint64, uint64]()
				// m.Reserve(int(size) / 4)
				start := time.Now()

				var wg sync.WaitGroup
				for i := range numGoros {
					wg.Add(1)
					goff := sizeg * uint64(i)
					go func() {
						defer wg.Done()
						var rng rand.PCG
						rng.Seed(rand.Uint64(), rand.Uint64())
						for range sizeg {
							k := goff + rng.Uint64()
							m.Put(k, k)
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*uint64(b.N))*float64(1000*1000)), "million_inserts/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelLookup(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		reps := (10 * 1000 * 1000) / size
		reps = max(1, reps)
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 4

			sizeg := size / uint64(numGoros)
			m := NewCon[uint64, uint64]()
			for k := uint64(0); k < size*50; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()

				for i := range numGoros {
					wg.Add(1)
					goff := sizeg * uint64(i)
					go func() {
						defer wg.Done()
						for range reps {
							for k := uint64(0); k < sizeg; k++ {
								v, _ := m.Get(k + goff)
								if v != k+goff {
									panic("bad")
								}
							}
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*reps*uint64(b.N)), "ns/lookup")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*reps*uint64(b.N))*float64(1000*1000)), "million_lookups/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelLookup2(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		reps := (10 * 1000 * 1000) / size
		reps = max(1, reps)
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 4

			sizeg := size / uint64(numGoros)
			m := NewCon2[uint64, uint64]()
			for k := uint64(0); k < size*50; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()

				for i := range numGoros {
					wg.Add(1)
					goff := sizeg * uint64(i)
					go func() {
						defer wg.Done()
						for range reps {
							for k := uint64(0); k < sizeg; k++ {
								v, _ := m.Get(k + goff)
								if v != k+goff {
									panic("bad")
								}
							}
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*reps*uint64(b.N)), "ns/lookup")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*reps*uint64(b.N))*float64(1000*1000)), "million_lookups/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelInsertBatches(b *testing.B) {
	for i := range sizes {
		size := sizes[i] * 10
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 4

			sizeg := size / uint64(numGoros)

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()
				m := NewCon[uint64, uint64]()

				for i := range numGoros {
					wg.Add(1)
					goff := sizeg * uint64(i)
					go func() {
						defer wg.Done()
						var batch [8]Kv[uint64, uint64]
						batch[0].Key = goff + 0
						batch[1].Key = goff + 1
						batch[2].Key = goff + 2
						batch[3].Key = goff + 3
						batch[4].Key = goff + 4
						batch[5].Key = goff + 5
						batch[6].Key = goff + 6
						batch[7].Key = goff + 7
						// var oks [8]bool
						// allOk := [...]bool{true, true, true, true, true, true, true, true}
						for k := uint64(0); k+8 < sizeg; k += 8 {
							m.PutBatch(&batch)
							// m.GetBatch(&batch, &oks)
							// if oks != allOk {
							// 	panic("bad")
							// }
							batch[0].Key += 8
							batch[1].Key += 8
							batch[2].Key += 8
							batch[3].Key += 8
							batch[4].Key += 8
							batch[5].Key += 8
							batch[6].Key += 8
							batch[7].Key += 8
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*uint64(b.N))*float64(1000*1000)), "million_inserts/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelInsertBatches2(b *testing.B) {
	for i := range sizes {
		size := sizes[i] * 10
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 4

			sizeg := size / uint64(numGoros)

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()
				m := NewCon2[uint64, uint64]()

				for i := range numGoros {
					wg.Add(1)
					goff := sizeg * uint64(i)
					go func() {
						defer wg.Done()
						var batch [8]Kv[uint64, uint64]
						batch[0].Key = goff + 0
						batch[1].Key = goff + 1
						batch[2].Key = goff + 2
						batch[3].Key = goff + 3
						batch[4].Key = goff + 4
						batch[5].Key = goff + 5
						batch[6].Key = goff + 6
						batch[7].Key = goff + 7
						// var oks [8]bool
						// allOk := [...]bool{true, true, true, true, true, true, true, true}
						for k := uint64(0); k+8 < sizeg; k += 8 {
							m.PutBatch(&batch)
							// m.GetBatch(&batch, &oks)
							// if oks != allOk {
							// 	panic("bad")
							// }
							batch[0].Key += 8
							batch[1].Key += 8
							batch[2].Key += 8
							batch[3].Key += 8
							batch[4].Key += 8
							batch[5].Key += 8
							batch[6].Key += 8
							batch[7].Key += 8
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*uint64(b.N))*float64(1000*1000)), "million_inserts/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelLookupBatches(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		reps := (10 * 1000 * 1000) / size
		reps = max(1, reps)
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 4

			sizeg := size / uint64(numGoros)
			m := NewCon[uint64, uint64]()
			for k := uint64(0); k < size*50; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()

				for i := range numGoros {
					wg.Add(1)
					goff := sizeg * uint64(i)
					go func() {
						defer wg.Done()
						for range reps {
							var batch [8]Kv[uint64, uint64]
							batch[0].Key = goff + 0
							batch[1].Key = goff + 1
							batch[2].Key = goff + 2
							batch[3].Key = goff + 3
							batch[4].Key = goff + 4
							batch[5].Key = goff + 5
							batch[6].Key = goff + 6
							batch[7].Key = goff + 7
							var oks [8]bool
							allOk := [...]bool{true, true, true, true, true, true, true, true}
							for k := uint64(0); k+8 < sizeg; k += 8 {
								m.GetBatch(&batch, &oks)
								if oks != allOk {
									panic("bad")
								}
								batch[0].Key += 8
								batch[1].Key += 8
								batch[2].Key += 8
								batch[3].Key += 8
								batch[4].Key += 8
								batch[5].Key += 8
								batch[6].Key += 8
								batch[7].Key += 8
							}
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*reps*uint64(b.N)), "ns/lookup")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*reps*uint64(b.N))*float64(1000*1000)), "million_lookups/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelLookupBatches2(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		reps := (10 * 1000 * 1000) / size
		reps = max(1, reps)
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 4

			sizeg := size / uint64(numGoros)
			m := NewCon2[uint64, uint64]()
			for k := uint64(0); k < size*50; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()

				for i := range numGoros {
					wg.Add(1)
					goff := sizeg * uint64(i)
					go func() {
						defer wg.Done()
						for range reps {
							var batch [8]Kv[uint64, uint64]
							batch[0].Key = goff + 0
							batch[1].Key = goff + 1
							batch[2].Key = goff + 2
							batch[3].Key = goff + 3
							batch[4].Key = goff + 4
							batch[5].Key = goff + 5
							batch[6].Key = goff + 6
							batch[7].Key = goff + 7
							var oks [8]bool
							allOk := [...]bool{true, true, true, true, true, true, true, true}
							for k := uint64(0); k+8 < sizeg; k += 8 {
								m.GetBatch(&batch, &oks)
								if oks != allOk {
									panic("bad")
								}
								batch[0].Key += 8
								batch[1].Key += 8
								batch[2].Key += 8
								batch[3].Key += 8
								batch[4].Key += 8
								batch[5].Key += 8
								batch[6].Key += 8
								batch[7].Key += 8
							}
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*reps*uint64(b.N)), "ns/lookup")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*reps*uint64(b.N))*float64(1000*1000)), "million_lookups/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelMixed(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 10

			sizeg := size / uint64(numGoros)
			sizeSmall := size / 10
			m := NewCon[uint64, uint64]()
			for k := uint64(0); k < sizeSmall; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()

				for range numGoros {
					wg.Add(1)
					go func() {
						defer wg.Done()
						rng := rand.NewPCG(rand.Uint64(), rand.Uint64())
						for range sizeg {
							what := rng.Uint64()
							action := what & (1<<10 - 1)

							if action < 30 { // delete likely
								what, _ = bits.Mul64(what, sizeSmall)
								m.Delete(what)
							} else if action < 200 {
								limit := size
								if action < 60 { // likely update
									limit = sizeSmall
								}
								what, _ = bits.Mul64(what, limit)
								m.Put(what, what)
							} else { // lookup likely hit
								limit := sizeSmall
								if action < 300 { // likely miss
									limit = size
								}
								what, _ = bits.Mul64(what, limit)
								m.Get(what)
							}
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/action")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*uint64(b.N))*float64(1000*1000)), "million_ops/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelMixed2(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 10

			sizeg := size / uint64(numGoros)
			sizeSmall := size / 10
			m := NewCon2[uint64, uint64]()
			for k := uint64(0); k < sizeSmall; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()

				for range numGoros {
					wg.Add(1)
					go func() {
						defer wg.Done()
						rng := rand.NewPCG(rand.Uint64(), rand.Uint64())
						for range sizeg {
							what := rng.Uint64()
							action := what & (1<<10 - 1)

							if action < 30 { // delete likely
								what, _ = bits.Mul64(what, sizeSmall)
								m.Delete(what)
							} else if action < 200 {
								limit := size
								if action < 60 { // likely update
									limit = sizeSmall
								}
								what, _ = bits.Mul64(what, limit)
								m.Put(what, what)
							} else { // lookup likely hit
								limit := sizeSmall
								if action < 300 { // likely miss
									limit = size
								}
								what, _ = bits.Mul64(what, limit)
								v, ok := m.Get(what)
								if ok && v != what {
									panic("bad")
								}
							}
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/action")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*uint64(b.N))*float64(1000*1000)), "million_ops/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConParallelMixed50502(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			numGoros := runtime.GOMAXPROCS(-1) * 10

			sizeg := size / uint64(numGoros)
			sizeSmall := size / 10
			m := NewCon2[uint64, uint64]()
			for k := uint64(0); k < sizeSmall; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				start := time.Now()

				for range numGoros {
					wg.Add(1)
					go func() {
						defer wg.Done()

						rng := rand.NewPCG(rand.Uint64(), rand.Uint64())
						for range sizeg {
							what := rng.Uint64()
							action := what & (1<<10 - 1)

							if action < 512 {
								limit := size
								what, _ = bits.Mul64(what, limit)
								m.Put(what, what)
							} else { // lookup likely hit
								limit := sizeSmall
								if action < 700 { // likely miss
									limit = size
								}
								what, _ = bits.Mul64(what, limit)
								v, ok := m.Get(what)
								if ok && v != what {
									panic("bad")
								}
							}
						}
					}()
				}
				wg.Wait()

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/action")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkConIterate(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := NewCon[uint64, uint64]()
			for k := uint64(0); k < size; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				start := time.Now()

				for range m.All {
				}

				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/lookup")
			b.ReportMetric(float64(time.Second)/(float64(b.Elapsed())/float64(size*uint64(b.N))*float64(1000*1000)), "million_lookups/s")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}
