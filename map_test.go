package fishtable

import (
	"fmt"
	"math"
	"math/bits"
	"runtime/debug"
	"testing"
	"time"
	"unsafe"
)

func init() {
	if true {
		debug.SetGCPercent(400 * 4)
	}
}

func TestInsert10m(t *testing.T) {
	m := New[uint64, uint64]()
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

func TestHashCollisions(t *testing.T) {
	m := New[uint64, uint64]()
	m.hasher = func(key unsafe.Pointer, seed uintptr) uintptr {
		return uintptr(*(*uint64)(key) % 32)
	}
	const c = 1 * 1000
	for k := uint64(0); k < c; k++ {
		m.Put(k, k)
	}
	for k := uint64(0); k < c; k++ {
		v, ok := m.Get(k)
		if !ok || v != k {
			println(ok, v, k)
			panic("bad")
		}
	}
}

func BenchmarkLoadFactor(b *testing.B) {
	var totalcap, totalpop uint64
	for i := 0; i < b.N; i++ {
		m := New[uint32, struct{}]()
		var stride int
		for k := uint32(0); k < 1*1000*1000; k++ {
			m.Put(k, struct{}{})
			if k&((1<<stride)-1) == 0 {
				stride = 32 - bits.LeadingZeros32(k/256)

				var capacity, pop uint64
				var chainDistribution [8]float64
				var totalSizeInBytes uint64
				var totalEntriesInBytes uint64
				m.iterateMaps(func(sm *bucketMap[uint32, struct{}]) {
					chainLength := 0
					for sm != nil {
						capacity += uint64(sm.cap)
						pop += uint64(sm.present.Count())
						chainDistribution[chainLength] += float64(sm.present.Count())

						totalSizeInBytes += uint64(unsafe.Sizeof(*sm))
						totalSizeInBytes += uint64(unsafe.Sizeof(sm.Kvs()[0])) * uint64(sm.cap)
						totalEntriesInBytes += uint64(unsafe.Sizeof(sm.Kvs()[0])) * uint64(sm.present.Count())

						chainLength++
						sm = sm.overflow
					}
				})

				totalSizeInBytes += uint64(unsafe.Sizeof(m.trie.Get(0))) * uint64(m.trie.Len())
				totalSizeInBytes += uint64(unsafe.Sizeof(*m))

				var lastNonZero int
				for i := range chainDistribution {
					if chainDistribution[i] != 0 {
						lastNonZero = i
					}
					chainDistribution[i] /= float64(m.len)
				}

				if true {
					fmt.Printf("%v:%v %.3f %.3f keys%%-in-nth-bucket=%.3f\n", pop, capacity, float64(pop)/float64(capacity), float64(totalEntriesInBytes)/float64(totalSizeInBytes), chainDistribution[:lastNonZero+1])
				}
				totalcap += capacity
				totalpop += pop
			}
		}
	}
	b.ReportMetric(float64(totalpop)/float64(totalcap), "lf/op")
}

var sizes = []uint64{
	1 * 10,
	1 * 90,
	1 * 420,
	1 * 1000,
	3 * 1000,
	10 * 1000,
	20 * 1000,
	50 * 1000,
	100 * 1000,
	300 * 1000,
	1000 * 1000,
	10 * 1000 * 1000,
}

func BenchmarkInsert(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				m := New[uint64, uint64]()
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

type s32 struct {
	n   uint64
	pad [3]uint64
}

func BenchmarkInsertStruct32(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				m := New[uint64, s32]()
				start := time.Now()
				for k := uint64(0); k < size; k++ {
					m.Put(k, s32{n: k})
				}
				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkInsertUint32Set(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				m := New[uint32, struct{}]()
				start := time.Now()
				for k := uint32(0); k < uint32(size); k++ {
					m.Put(k, struct{}{})
				}
				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

const numColdMaps = 64 // realistic?

func BenchmarkCold(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			var mind time.Duration = math.MaxInt64
			maps := make([]*Map[uint64, uint64], numColdMaps)
			for i := 0; i < b.N; i++ {
				for j := range maps {
					maps[j] = New[uint64, uint64]()
				}
				start := time.Now()
				for k := uint64(0); k < size*numColdMaps; k++ {
					m := maps[k%numColdMaps]
					m.Put(k, k)
				}
				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*numColdMaps*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}

func BenchmarkSizedInsert(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				m := New[uint64, uint64]()
				m.Reserve(int(size))
				for k := uint64(0); k < size; k++ {
					m.Put(k, k)
				}
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
		})
	}
}

func BenchmarkInsertReuse(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := New[uint64, uint64]()
			for k := uint64(0); k < size; k++ {
				m.Put(k, k)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for k := uint64(0); k < size; k++ {
					m.Delete(k)
				}
				for k := uint64(0); k < size; k++ {
					m.Put(k, k)
				}
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
		})
	}
}

func BenchmarkDelete(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				m := New[uint64, uint64]()
				for k := uint64(0); k < size; k++ {
					m.Put(k, k)
				}
				b.StartTimer()
				for k := uint64(0); k < size; k++ {
					m.Delete(k)
				}
				for k := uint64(0); k < size; k++ {
					m.Delete(k)
				}
				for k := uint64(0); k < size; k++ {
					m.Delete(k)
				}
				for k := uint64(0); k < size; k++ {
					m.Delete(k)
				}
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)*4), "ns/delete")
		})
	}
}

func BenchmarkClear(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := New[uint64, uint64]()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for k := uint64(0); k < size; k++ {
					m.Put(k, k)
				}
				m.Clear()
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
		})
	}
}

func BenchmarkLookup(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := New[uint64, uint64]()
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

func BenchmarkLookupsStruct32(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := New[uint64, s32]()
			for k := uint64(0); k < size; k++ {
				m.Put(k, s32{n: k})
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for k := uint64(0); k < size; k++ {
					v, _ := m.Get(k)
					if v.n != k {
						panic("bad")
					}
				}
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/lookup")
		})
	}
}

func BenchmarkMisses(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := New[uint64, uint64]()
			for k := uint64(0); k < size; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for k := uint64(0); k < size; k++ {
					hit := k&1 == 1
					add := uint64(0)
					if !hit {
						add = size
					}
					_, ok := m.Get(k + add)
					if ok != hit {
						panic("bad")
					}
				}
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/lookup")
		})
	}
}

func BenchmarkIter(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := New[uint64, uint64]()
			for k := uint64(0); k < size; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.All(func(k, v uint64) bool { return true })
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/lookup")
		})
	}
}

func BenchmarkStringLookups(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := New[string, uint64]()
			strings := make([]string, size)
			// fmtstr := "%" + fmt.Sprintf("0%vv", 1+int(math.Log10(float64(size))))
			fmtstr := "%" + fmt.Sprintf("0%vv", 1+int(math.Log10(float64(size))*4))
			for k := uint64(0); k < size; k++ {
				strings[k] = fmt.Sprintf(fmtstr, k)
				m.Put(strings[k], k)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for k := uint64(0); k < size; k++ {
					v, _ := m.Get(strings[k])
					if v != k {
						panic("bad")
					}
				}
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/lookup")
		})
	}
}

func dumpPhDepths[K comparable, V any](m *Map[K, V]) {
	var depths [64 + 1]int
	var ovfs [64 + 1]int
	var capacity, popcount uint64
	var overflows, longestOverflow int
	var bb, bb2 uint64
	var totalMaps int
	var countsBySize [bucketSize + 1]int
	var countsBySize2 [bucketSize + 1]int
	var countsByCap [bucketSize + 1]int
	m.iterateMaps(func(sm *bucketMap[K, V]) {
		depth := sm.depth

		var chain int
		for sm != nil {
			capacity += uint64(sm.cap)
			countsByCap[sm.cap]++
			popcount += uint64(sm.present.Count())
			bb += uint64(unsafe.Sizeof(*sm.first())) * uint64(sm.present.Count())
			bb2 += uint64(unsafe.Sizeof(*sm.first())) * uint64(sm.cap)
			bb2 += uint64(unsafe.Sizeof(*sm))
			totalMaps++
			if chain == 0 {
				countsBySize[sm.present.Count()]++
			} else {
				countsBySize2[sm.present.Count()]++
			}
			if sm.overflow != nil {
				overflows++
				chain++
			}

			sm = sm.overflow
		}
		longestOverflow = max(longestOverflow, chain)

		depths[depth]++
		ovfs[depth] += chain
	})
	bb2 += uint64(unsafe.Sizeof(*m.trie.first)) * uint64(m.trie.Len())

	var maxDepth int
	var minDepth int = 64
	for depth, cnt := range depths {
		if cnt != 0 {
			maxDepth = max(maxDepth, depth)
			minDepth = min(minDepth, depth)
		}
	}

	fmt.Printf("%v:%v:	lf:%0.3f	overflows:%v	longest-overflow-chain: %v\n", popcount, capacity, float64(popcount)/float64(capacity), overflows, longestOverflow)
	fmt.Printf("%v %v\n", m.len, totalMaps)
	fmt.Printf("%v:%v:	memlf:%0.3f\n", bb, bb2, float64(bb)/float64(bb2))

	fmt.Printf("depths:	level	maps\n")
	fmt.Printf("	[..]	0\n")
	for i := minDepth; i <= maxDepth; i++ {
		fmt.Printf("	[%02v]	%v	has-overflow=%v\n", i, depths[i], ovfs[i])
	}
	fmt.Printf("	[..]	0\n")

	fmt.Printf("\nBucket stats:\n")
	fmt.Printf("	pop	maps	overflow maps	maps by cap\n")
	for i, cnt := range countsBySize {
		fmt.Printf("	[%02v]	%v	%v		%v\n", i, cnt, countsBySize2[i], countsByCap[i])
	}
}
