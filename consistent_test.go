package fishtable

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestConsistentInsert10m(t *testing.T) {
	m := NewConsistent[uint64, uint64]()
	const c = 10 * 1000 * 1000
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
			kv, v, ok := m.GetClosest(k)
			if !ok || kv != k || v != k {
				t.Fatalf("%v %v %v %v", ok, k, kv, v)
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

func BenchmarkConsistentInsert(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				m := NewConsistent[uint64, uint64]()
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

func BenchmarkConsistentLookup(b *testing.B) {
	for i := range sizes {
		size := sizes[i]
		b.Run(fmt.Sprintf("size=%v", size), func(b *testing.B) {
			m := NewConsistent[uint64, uint64]()
			for k := uint64(0); k < size; k++ {
				m.Put(k, k)
			}

			b.ResetTimer()

			var mind time.Duration = math.MaxInt64
			for i := 0; i < b.N; i++ {
				start := time.Now()
				for k := uint64(0); k < size; k++ {
					k2 := k * 11400714819323198487
					_, _, ok := m.GetClosest(k2)
					if !ok {
						panic("bad")
					}
				}
				mind = min(mind, time.Since(start))
			}

			b.ReportMetric(float64(b.Elapsed())/float64(size*uint64(b.N)), "ns/insert")
			b.ReportMetric(float64(mind), "ns/min")
		})
	}
}
