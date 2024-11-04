package fishtable

import (
	"math/bits"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

func NewCon2[K comparable, V any]() *ConMap2[K, V] {
	m := &ConMap2[K, V]{
		hasher: getRuntimeHasher[K](),
		seed:   uintptr(runtime_fastrand64()),
	}
	m.newTrie.Update(sarray[te[K, V]]{shift: 64})
	m.oldTrie.Update(sarray[te[K, V]]{shift: 64})
	return m
}

type ConMap2[K comparable, V any] struct {
	// The newest trie. As one of the goroutines works on populating this trie with
	// trie entries, the others may in parallel access and update this trie.
	//
	// If resizing is not taking place it just points
	// to the exactly same trie as 'trie' does. But if resizing is taking
	// place then all bucket splits made during the resize are stored
	// correctly only here.
	newTrie consarray[te[K, V]]

	// the trie that is always fully populated with good buckets.
	//
	// After a bucket splits then the other half might not be in here but can be
	// found in 'newTrie'.
	oldTrie consarray[te[K, V]]

	// resizing state of the trie
	resizing atomic.Pointer[resizingState[te[K, V]]]

	hasher func(key unsafe.Pointer, seed uintptr) uintptr
	seed   uintptr

	pad [128 - 40]byte
	len atomic.Int64

	// Real languages have bulk free() so freeing all of the the little maps
	// should be fast.
}

type conBucketMap2[K comparable, V any] struct {
	mu sync.Mutex

	// For Optimistic concurrency for readers.
	version atomic.Uint64

	fingerprints [bucketSize]uint8
	present      bucketBitmask
	depth        uint8 // <= 64, also used as the timestamp value in the trie
	overflow     *conBucketMap2[K, V]

	// TODO: Align bucketSize so that for K and V it eats up a nice number of cache-lines
	// with little unused space. Might have to adjust to range [16, 32).
	kvs [bucketSize]Kv[K, V]
}

func (t *te[K, V]) Load2() (*conBucketMap2[K, V], uint64) {
	packed := atomic.LoadPointer(&t.ptr)
	m := (*conBucketMap2[K, V])(unsafe.Pointer(uintptr(packed) &^ 0b11_1111))
	shift := uint64(uintptr(packed) & 0b11_1111)
	return m, shift
}

func (t *te[K, V]) Store2(m *conBucketMap2[K, V], ts uint64) {
	// unsafe.Pointer value is legal as long as it points _somewhere_ into the object. Because
	// our buckets always have sizeof >= 64B (and alignment of 64) it means that we can pack the
	// 6-bit value into the lowest 6 bits so that the resulting pointer will continue to point into
	// the original allocated object.
	packed := unsafe.Add(unsafe.Pointer(m), ts)
	atomic.StorePointer(&t.ptr, packed)
}

func conTrieGet2[K comparable, V any](a sarray[te[K, V]], i uintptr) (*conBucketMap2[K, V], uint64) {
	trieEntry := (*te[K, V])(unsafe.Add(unsafe.Pointer(a.first), unsafe.Sizeof(*a.first)*i))
	return trieEntry.Load2()
}

func conTrieGetPacked2[K comparable, V any](a sarray[te[K, V]], i uintptr) unsafe.Pointer {
	trieEntry := (*te[K, V])(unsafe.Add(unsafe.Pointer(a.first), unsafe.Sizeof(*a.first)*i))
	return atomic.LoadPointer(&trieEntry.ptr)
}

func conTrieSet2[K comparable, V any](a sarray[te[K, V]], i uintptr, v *conBucketMap2[K, V], ts uint64) {
	trieEntry := (*te[K, V])(unsafe.Add(unsafe.Pointer(a.first), unsafe.Sizeof(*a.first)*i))
	// inlined trieEntry.Store(v, ts) so that conTrieSet itself inlines
	packed := unsafe.Add(unsafe.Pointer(v), ts)
	atomic.StorePointer(&trieEntry.ptr, packed)
}

func conUnpackPacked2[K comparable, V any](packed unsafe.Pointer) (*conBucketMap2[K, V], uint64) {
	m := (*conBucketMap2[K, V])(unsafe.Pointer(uintptr(packed) &^ 0b11_1111))
	shift := uint64(uintptr(packed) & 0b11_1111)
	return m, shift
}

func (m *ConMap2[K, V]) getLockedBucket(trie sarray[te[K, V]], hash uintptr) (sarray[te[K, V]], *conBucketMap2[K, V]) {
	for {
		trieEntry, trieEntryTs := conTrieGet2(trie, hash>>(trie.shift%64))
		if trieEntry == nil {
			// Only newtrie can return nils and so only if it's in the middle of the resizing process and hasn't
			// yet moved in the buckets for this hash.
			//
			// Look at oldTrie as it's guaranteed to contain something for this hash.
			// Assume no bugs.
			trie = m.oldTrie.Load()
			continue
		}

		trieEntry.mu.Lock() // lock elision would be nice here

		// The bucket may have split while we tried to Lock it.
		if trieEntry.depth == uint8(trieEntryTs) {
			return trie, trieEntry
		}

		// Did we get lucky and pick get the correct bucket anyways? There's a ~50% chance for it!
		trie = m.newTrie.Load()
		trieEntry2, _ := conTrieGet2(trie, hash>>(trie.shift%64))
		if trieEntry == trieEntry2 {
			return trie, trieEntry
		}

		trieEntry.mu.Unlock()
	}
}

func (m *ConMap2[K, V]) getUnlockedBucket(trie sarray[te[K, V]], hash uintptr) (*conBucketMap2[K, V], uint64) {
	trieEntry, ts0 := conTrieGet2(trie, hash>>(trie.shift%64))
	if trieEntry == nil {
		// Only newtrie can return nils and so only if it's in the middle of the resizing process and hasn't
		// yet moved in the buckets for this hash.
		//
		// Look at oldTrie as it's guaranteed to contain something for this hash.
		// Assume no bugs.
		trie = m.oldTrie.Load()
		trieEntry, ts0 = conTrieGet2(trie, hash>>(trie.shift%64))
	}

	v0 := trieEntry.version.Load()

	// The bucket may have split before we loaded the version
	if !raceEnabled && ts0 != uint64(trieEntry.depth) {
		v0 = 1 // force the user to use a mutex for the bucket
	}

	return trieEntry, v0
}

func (b *conBucketMap2[K, V]) GetKv(i int) *Kv[K, V] {
	return &b.kvs[i]
}

func (b *conBucketMap2[K, V]) SetKv(i int, v Kv[K, V]) {
	b.kvs[i] = v
}

func (b *conBucketMap2[K, V]) Kvs() []Kv[K, V] {
	return b.kvs[:]
}

func newConBucketMap2[K comparable, V any]() *conBucketMap2[K, V] {
	bucket := new(conBucketMap2[K, V])
	if unsafe.Sizeof(*bucket) < 64 {
		panic("Key and Value types must be large enough to make the buckets at least 64 bytes big.")
	}

	// We require alignment of 64 and size of 64. Check that the allocator gives it to us.
	// TODO: https://github.com/golang/go/issues/19057
	ptr := uintptr(unsafe.Pointer(bucket))
	if ptr&0b11_1111 != 0 {
		println("ptr =", ptr)
		panic("Key and Value types must be large enough to fool the allocator to align buckets to 64 byte boundaries.")
	}
	return bucket
}

func (m *ConMap2[K, V]) Get(k K) (v V, ok bool) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)
	fingerprint := fixFingerprint(uint64(hash))
	probe := makeHashProbe(fingerprint)

	trie := m.newTrie.Load()
	if trie.shift >= 64 {
		return
	}

	trieEntry, v0 := m.getUnlockedBucket(trie, hash)

	if v0&1 == 0 && !raceEnabled {
		// lock-free search
		for bucket := trieEntry; bucket != nil; bucket = bucket.overflow {
			finder := bucketFinderFrom(&bucket.fingerprints)

			matches := finder.ProbeHashMatches(probe)
			matches = matches.Keep(bucket.present)
			for ; matches.HasCurrent(); matches.Advance() {
				slotInBucket := matches.Current()

				// Copy the KV value to stack so that we can ensure
				// that it's not a torn write that could cause havoc
				// for the eq check. String keys could be spooky if torn.
				kvcopy := *bucket.GetKv(slotInBucket)

				// Did writer write and kvcopy is possibly torn?
				if v0 != trieEntry.version.Load() {
					// Writer wrote a new version. Retry the read while holding
					// the mutex.
					goto withMutex
				}

				if kvcopy.Key == k {
					v = kvcopy.Value
					ok = true
					return
				}
			}
		}

		// If we did not find the key it could be because of a torn read of the bucket state
		if v0 == trieEntry.version.Load() {
			return // All good!
		}

		// Writer wrote a new version. Retry the read while holding
		// the mutex.
	}

withMutex:
	_, trieEntry = m.getLockedBucket(trie, hash)

bucketLoop:
	for bucket := trieEntry; bucket != nil; bucket = bucket.overflow {
		finder := bucketFinderFrom(&bucket.fingerprints)

		matches := finder.ProbeHashMatches(probe)
		matches = matches.Keep(bucket.present)
		for ; matches.HasCurrent(); matches.Advance() {
			slotInBucket := matches.Current()
			kv := bucket.GetKv(slotInBucket)
			if kv.Key == k {
				v = kv.Value
				ok = true
				break bucketLoop
			}
		}
	}

	trieEntry.mu.Unlock()

	return
}

func (m *ConMap2[K, V]) GetBatch(kvs *[8]Kv[K, V], oks *[8]bool) {
	clear(oks[:]) // set all to false

	trie := m.newTrie.Load()
	if trie.shift >= 64 {
		return
	}

	var hashes [8]uintptr
	for i := range kvs {
		hash := m.hasher(noescape(unsafe.Pointer(&kvs[i].Key)), m.seed)
		hashes[i] = hash
	}

	// Prefetch
	// for i := range kvs {
	// 	hash := hashes[i]
	// 	packed := conTrieGetPacked2(trie, hash>>(trie.shift%64)) // cache miss #1
	// 	runtime.Prefetch(uintptr(packed))                        // cache miss #2
	// }

kvloop:
	for i := range kvs {
		hash := hashes[i]
		fingerprint := fixFingerprint(uint64(hash))
		probe := makeHashProbe(fingerprint)

		trieEntry, v0 := m.getUnlockedBucket(trie, hash)

		if v0&1 == 0 && !raceEnabled {
			// lock-free search
			for bucket := trieEntry; bucket != nil; bucket = bucket.overflow {
				finder := bucketFinderFrom(&bucket.fingerprints)

				matches := finder.ProbeHashMatches(probe)
				matches = matches.Keep(bucket.present)
				for ; matches.HasCurrent(); matches.Advance() {
					slotInBucket := matches.Current()

					// Copy the KV value to stack so that we can ensure
					// that it's not a torn write that could cause havoc
					// for the eq check. String keys could be spooky if torn.
					kvcopy := *bucket.GetKv(slotInBucket)

					// Did writer write and kvcopy is possibly torn?
					if v0 != trieEntry.version.Load() {
						// Writer wrote a new version. Retry the read while holding
						// the mutex.
						goto withMutex
					}

					if kvcopy.Key == kvs[i].Key {
						kvs[i].Value = kvcopy.Value
						oks[i] = true
						continue kvloop
					}
				}
			}

			// If we did not find the key it could be because of a torn read of the bucket state
			if v0 == trieEntry.version.Load() {
				continue kvloop // All good!
			}

			// Writer wrote a new version. Retry the read while holding
			// the mutex.
		}

	withMutex:
		_, trieEntry = m.getLockedBucket(trie, hash)

	bucketLoop:
		for bucket := trieEntry; bucket != nil; bucket = bucket.overflow {
			finder := bucketFinderFrom(&bucket.fingerprints)

			matches := finder.ProbeHashMatches(probe)
			matches = matches.Keep(bucket.present)
			for ; matches.HasCurrent(); matches.Advance() {
				slotInBucket := matches.Current()
				kv := bucket.GetKv(slotInBucket)
				if kv.Key == kvs[i].Key {
					kvs[i].Value = kv.Value
					oks[i] = true
					break bucketLoop
				}
			}
		}

		trieEntry.mu.Unlock()
	}
	return
}

func (m *ConMap2[K, V]) Delete(k K) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)
	fingerprint := fixFingerprint(uint64(hash))
	probe := makeHashProbe(fingerprint)

	trie := m.newTrie.Load()
	if trie.shift >= 64 {
		return
	}

	_, trieEntry := m.getLockedBucket(trie, hash)

	for bucket := trieEntry; bucket != nil; bucket = bucket.overflow {
		finder := bucketFinderFrom(&bucket.fingerprints)

		matches := finder.ProbeHashMatches(probe)
		matches = matches.Keep(bucket.present)
		for ; matches.HasCurrent(); matches.Advance() {
			slotInBucket := matches.Current()
			kv := bucket.GetKv(slotInBucket)
			if kv.Key == k {
				trieEntry.version.Add(1)
				*kv = Kv[K, V]{} // for GC
				bucket.present.Unmark(slotInBucket)
				trieEntry.version.Add(1)
				m.len.Add(-1)
				// NOTE: Releasing empty overflow buckets is deferred until the next Put()
				// into this trie bucket
				goto exit
			}
		}
	}

exit:
	trieEntry.mu.Unlock()
	return
}

func (m *ConMap2[K, V]) Put(k K, v V) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)
	fingerprint := fixFingerprint(uint64(hash))
	probe := makeHashProbe(fingerprint)

	trie := m.newTrie.Load()

	var trieEntry, bucket *conBucketMap2[K, V]
	if trie.shift < 64 {
		trie, trieEntry = m.getLockedBucket(trie, hash)
		bucket = trieEntry
		if trieEntry == nil {
			panic("bad")
		}
	}

	for {
		// bucket is nil for empty maps
		if bucket != nil {
			finder := bucketFinderFrom(&bucket.fingerprints)

			// Any matching hashes?
			matches := finder.ProbeHashMatches(probe)
			matches = matches.Keep(bucket.present) // is there an entry actually present there?
			for ; matches.HasCurrent(); matches.Advance() {
				slotInBucket := matches.Current()
				if kv := bucket.GetKv(slotInBucket); kv.Key == k {
					trieEntry.version.Add(1)
					kv.Value = v
					trieEntry.version.Add(1)
					trieEntry.mu.Unlock()
					return
				}
			}

			// If we reached the end of the linked list we know that the Key is not in this map.
			// We can try inserting then if there's free space to do so.
			if bucket.overflow == nil {
				nextFree := bucket.present.FirstUnmarkedSlot()
				if nextFree < bucketSize {
					trieEntry.version.Add(1)
					bucket.SetKv(nextFree, Kv[K, V]{Key: k, Value: v})
					bucket.fingerprints[nextFree] = uint8(fingerprint)
					bucket.present.Mark(nextFree)
					trieEntry.version.Add(1)
					m.len.Add(1)
					trieEntry.mu.Unlock()
					return
				}
			}
		}

		trie, trieEntry, bucket = m.makeSpaceForHash(trie, trieEntry, bucket, hash)
	}
}

func (m *ConMap2[K, V]) PutBatch(kvs *[8]Kv[K, V]) {
	var hashes [8]uintptr
	for i := range kvs {
		hash := m.hasher(noescape(unsafe.Pointer(&kvs[i].Key)), m.seed)
		hashes[i] = hash
	}

	trie := m.newTrie.Load()
	// Prefetch
	// if trie.shift < 64 {
	// 	for i := range kvs {
	// 		hash := hashes[i]
	// 		packed := conTrieGetPacked(trie, hash>>(trie.shift%64)) // cache miss #1
	// 		runtime.Prefetch(uintptr(packed))                       // cache miss #2
	// 	}
	// }

kvsLoop:
	for i := range kvs {
		hash := hashes[i]
		fingerprint := fixFingerprint(uint64(hash))
		probe := makeHashProbe(fingerprint)

		var trieEntry, bucket *conBucketMap2[K, V]
		if trie.shift < 64 {
			_, trieEntry = m.getLockedBucket(trie, hash)
			bucket = trieEntry
		}

		for {
			// bucket is nil for empty maps
			if bucket != nil {
				finder := bucketFinderFrom(&bucket.fingerprints)

				// Any matching hashes?
				matches := finder.ProbeHashMatches(probe)
				matches = matches.Keep(bucket.present) // is there an entry actually present there?
				for ; matches.HasCurrent(); matches.Advance() {
					slotInBucket := matches.Current()
					if kv := bucket.GetKv(slotInBucket); kv.Key == kvs[i].Key {
						trieEntry.version.Add(1)
						kv.Value = kvs[i].Value
						trieEntry.version.Add(1)
						trieEntry.mu.Unlock()
						continue kvsLoop
					}
				}

				// If we reached the end of the linked list we know that the Key is not in this map.
				// We can try inserting then if there's free space to do so.
				if bucket.overflow == nil {
					nextFree := bucket.present.FirstUnmarkedSlot()
					if nextFree < bucketSize {
						trieEntry.version.Add(1)
						bucket.SetKv(nextFree, kvs[i])
						bucket.fingerprints[nextFree] = uint8(fingerprint)
						bucket.present.Mark(nextFree)
						trieEntry.version.Add(1)
						m.len.Add(1)
						trieEntry.mu.Unlock()
						continue kvsLoop
					}
				}
			}

			trie, trieEntry, bucket = m.makeSpaceForHash(trie, trieEntry, bucket, hash)
		}
	}
}

// figures out a good bucket for an entry with the given hash.
func (m *ConMap2[K, V]) makeSpaceForHash(trie sarray[te[K, V]], trieEntry, bucket *conBucketMap2[K, V], hash uintptr) (sarray[te[K, V]], *conBucketMap2[K, V], *conBucketMap2[K, V]) {
	// bucket/map0 is nil if the map is empty
	if bucket == nil {
		trie = m.resizeAtLeastForDepth(1)

		// return the updated map from the trie
		trie, trieEntry := m.getLockedBucket(trie, hash)
		return trie, trieEntry, trieEntry
	}

	// Are we still in the lookup phase trying to just check if the linked list of buckets contains the key
	if bucket.overflow != nil {
		// Also use this as an opportunity to remove useless overflow buckets.
		//
		// After doing many deletes we may have ended up with useless overflow buckets. Unless the hash function is shit
		// the linked list of buckets isn't longer than 2 buckets (initial + overflow1).
		if bucket.present.IsAllSet() {
			// Nothing to do, continue to the overflow
			return trie, trieEntry, bucket.overflow
		}

		// Could the overflow bucket fit into this bucket?
		freeSpace := bucketSize - bucket.present.Count()
		if freeSpace < bucket.overflow.present.Count() {
			// Not enough free space to merge the two buckets. Continue to overflow
			return trie, trieEntry, bucket.overflow
		}

		trieEntry.version.Add(1)

		// Move the overflow bucket's contents into this bucket
		for presents := bucket.overflow.present.AsMatchiter(); presents.HasCurrent(); presents.Advance() {
			slotInBucket := presents.Current()

			nextFree := bucket.present.FirstUnmarkedSlot()
			if nextFree >= bucketSize {
				panic("bad")
			}

			bucket.SetKv(nextFree, *bucket.overflow.GetKv(slotInBucket))
			bucket.fingerprints[nextFree] = bucket.overflow.fingerprints[slotInBucket]
			bucket.present.Mark(nextFree)
		}

		// Drop the overflow
		bucket.overflow = bucket.overflow.overflow

		trieEntry.version.Add(1)

		// Retry looking from the bucket as it may now contain the key
		// that the user is trying to update
		return trie, trieEntry, bucket
	}

	globalDepth := (64 - trie.shift) % 64

	// Allocate overflow if we would cause the trie to grow too much
	// TODO: We could make it so that if we are about to allocate the 4th entry to the overflow chain
	// then we just split(). This would give us the property of being a perfect hash table because in the worst
	// case lookups would need to look at most into 3 buckets thus making it O(1) = perfect hash table
	if trie.shift < 64 && uintptr(bucket.depth) >= globalDepth {
		// How big should the trie be considering the total number of entries we are storing?
		// NOTE: This here is basically _the_ load-factor setting of this hash map
		// NOTE: The idea here is that as the map grows to a larger size (in the tens of thousands) we start to
		// aim for a higher load-factor to keep more of our data in our cpu caches and thus we are faster.
		// As expected this will cause our linked list of overflow buckets to grow longer and longer
		// but when 99.999% of entries can be reached through the first 2 buckets it doesn't matter
		// that we have 3rd and 4th overflow buckets. It's exceptionally rare to even have the 4th
		// bucket and I have never seen a 5th bucket.
		len := int(m.len.Load())
		expectedTrieSize := len/bucketSize + 32<<(bits.Len(uint(len))/2)
		if trie.Len() >= expectedTrieSize {
			if !bucket.present.IsAllSet() {
				panic("bad")
			}

			// Allocate overflow
			trieEntry.version.Add(1)
			bucket.overflow = newConBucketMap2[K, V]()
			bucket.overflow.depth = bucket.depth
			trieEntry.version.Add(1)

			return trie, trieEntry, bucket.overflow
		}
	}

	// Before inserting the new bucket to the trie check if the trie needs to grow as well...
	if requiredDepth := uintptr(trieEntry.depth + 1); requiredDepth > globalDepth {
		trieEntry.mu.Unlock() // release the bucket lock so any Updates can run as we resize

		trie = m.resizeAtLeastForDepth(uint8(requiredDepth))

		// return the updated map from the trie and force the Put loop to try again
		// from the beginning
		trie, trieEntry := m.getLockedBucket(trie, hash)
		return trie, trieEntry, trieEntry
	}

	// If resizing is in progress we should help complete it. We don't want to be freeloaders
	// that use structures that others have worked hard to build.
	if globalDepth >= uintptr(bits.Len(64*resizingChunkSize)) {
		friendResizing := m.resizing.Load()
		if friendResizing != nil {
			trieEntry.mu.Unlock() // release the bucket lock so any Updates can run as we resize

			m.helpWithCompletingResizing(friendResizing)

			// return the updated map from the trie and force the Put loop to try again
			// from the beginning
			trie, trieEntry := m.getLockedBucket(friendResizing.newTrie, hash)
			return trie, trieEntry, trieEntry
		}
	}

	// split it
	trieEntry.version.Add(1)
	trieEntry.depth++

	// Do the split moves
	newBucket, swapped := m.split(trieEntry)

	// Take the lock for the new bucket so that both "siblings" are locked. The new bucket
	// is not yet visible to other threads so it can't be locked yet. When we are
	// done with updating the trie and reinserting the overflow entries then
	// we will release the lock for the bucket that we no longer need for the
	// current hash.
	newBucket.mu.Lock()
	newBucket.version.Add(1)

	// Insert "newBucket" into the trie and update timestamp for original bucket
	leftb, rightb := maybeSwap(trieEntry, newBucket, swapped)
	for {
		shift := trie.shift % 64
		oldCount := uintptr(1) << (globalDepth - uintptr(newBucket.depth-1))
		startpos := uintptr(hash>>shift) &^ (oldCount - 1)
		if oldCount < 2 {
			panic("bad")
		}

		// TODO: 99% the time (oldCount/2) is <= 2. Do something with it.

		for i := startpos; i < startpos+oldCount/2; i++ {
			conTrieSet2(trie, i, leftb, uint64(leftb.depth))
		}

		startpos += (oldCount / 2)
		for i := startpos; i < startpos+oldCount/2; i++ {
			conTrieSet2(trie, i, rightb, uint64(rightb.depth))
		}

		// Another goroutine in the background could have switched us to a new trie array.
		// In that case we need to switch to that trie array and repeat the insert into
		// it.
		newTrie := m.newTrie.Load()
		if trie == newTrie {
			break
		}

		// refresh the current view of the trie and do it again
		trie = newTrie
		globalDepth = (64 - trie.shift) % 64
	}

	{
		// The overflow entries from the original first bucket need to be split
		// too.
		overflow := trieEntry.overflow
		if overflow != nil {
			trieEntry.overflow = nil
			m.reinsertOverflowBuckets(leftb, rightb, overflow)
		}
	}

	trieEntry.version.Add(1)
	newBucket.version.Add(1)

	// return the correct bucket of the two for the hash
	chosimba, loser := pickForHash(leftb, rightb, hash, leftb.depth)
	loser.mu.Unlock()
	return trie, chosimba, chosimba
}

// reinsertOverflowBucket inserts the entries from an overflow bucket into the map
func (m *ConMap2[K, V]) reinsertOverflowBuckets(leftb, rightb, firstOverflow *conBucketMap2[K, V]) {
	// TODO: When we start to have (N > 100k) entries some of our overflow buckets
	// have so many entries that the main buckets rarely have enough room, requiring
	// overflow buckets to be allocated. The situation just gets worse as we start
	// to have millions of entries. Optimize this function with that in mind by
	// trying to re-use the given overflow bucket as the new overflow bucket and not
	// doing any moves unless necessary. Consider that at ~200k entries our overflow
	// bucket starts to have an overflow bucket of its own.
	next := firstOverflow
	for overflow := firstOverflow; overflow != nil; overflow = next {
		next = overflow.overflow
		overflow.overflow = nil

		for presents := overflow.present.AsMatchiter(); presents.HasCurrent(); presents.Advance() {
			slotInBucket := presents.Current()

			kv := overflow.GetKv(slotInBucket)
			hash := m.hasher(unsafe.Pointer(&kv.Key), m.seed)
			fingerprint := fixFingerprint(uint64(hash))

			bucket, _ := pickForHash(leftb, rightb, hash, leftb.depth)
			for {
				// We can try inserting then if there's free space to do so. Unlike in Put() we know that the later buckets
				// in the linked list can't already contain this key.
				nextFree := bucket.present.FirstUnmarkedSlot()
				if nextFree < bucketSize {
					bucket.SetKv(nextFree, *kv)
					bucket.fingerprints[nextFree] = uint8(fingerprint)
					bucket.present.Mark(nextFree)
					break
				}
				if bucket.overflow == nil {
					bucket.overflow = newConBucketMap2[K, V]()
					bucket.overflow.depth = bucket.depth
				}

				// make leftb and rightb point to the last entry in their linked list so we don't have to always
				// find it again.
				if bucket == leftb {
					leftb = bucket.overflow
				} else {
					rightb = bucket.overflow
				}
				bucket = bucket.overflow
			}
		}
	}
}

func (h *ConMap2[K, V]) split(left *conBucketMap2[K, V]) (*conBucketMap2[K, V], bool) {
	if left.depth >= 64 || left.depth <= 0 {
		// 64-bit hash values have only 64 bits :-(
		panic("depth overflow")
	}

	oldDepth := left.depth - 1
	hashShift := (63 - oldDepth) % 64

	// TODO: Prefetch the keys before hashing them?

	// Move righties to the right map
	// NOTE: Doing it in 2 steps is much faster. One step version would have just a single
	// loop that has a branch in the body that checks if the entry should go to the right.
	// But it causes too many branch-misses.

	// Figure out which slots contain entries that will be moved to the "right" map
	var rightGoers bucketBitmask

	// NOTE: We know that the bucket is always full when we split()
	kvs := left.Kvs()
	for slotInBucket := range kvs {
		hash := h.hasher(unsafe.Pointer(&kvs[slotInBucket].Key), h.seed)

		goesRight := (hash >> hashShift) & 1
		rightGoers.MaybeMark(slotInBucket, goesRight)
	}

	// We swap left and right around so that we minimize the amount of moves that we need to do.
	var swap bool
	if cnt := rightGoers.Count(); left.present.Count()-cnt < cnt {
		swap = true
	}

	if swap {
		old := rightGoers
		rightGoers = left.present
		rightGoers.UnmarkAll(old)
	}

	// allocate the right map
	// NOTE: There is a way to write this function so that we can re-use the overflow bucket
	// as the 'right' bucket, avoiding unnecessarily allocating/freeing in such case. But
	// it's such a mess to write that I don't want to do it.
	right := newConBucketMap2[K, V]()
	right.depth = left.depth

	// Do the moving

	// Outside the loop with SIMD
	left.present.UnmarkAll(rightGoers)
	right.present.MarkFirst(rightGoers.Count())
	// TODO: Move tophash8 with SIMD

	// NOTE: We fill from array start so that more of our kvs share their cache lines. Speeds up lookups
	// NOTE: For the "left" map too we could move the remaining entries to the start so that
	// more of them are loaded in as part of the initial cache-line.
	var nextFreeInRight int
	for presents := rightGoers.AsMatchiter(); presents.HasCurrent(); presents.Advance() {
		slotInBucket := presents.Current()

		lhs := left.GetKv(slotInBucket)
		right.SetKv(nextFreeInRight, *lhs)
		right.fingerprints[nextFreeInRight] = left.fingerprints[slotInBucket]
		nextFreeInRight++

		*lhs = Kv[K, V]{} // for the GC, release pointers
	}

	return right, swap
}

// resizes the trie so that a bucket at the target depth may be placed into it.
//
// As a special case if the current trie is empty it also allocates the initial
// bucket into it.
func (m *ConMap2[K, V]) resizeAtLeastForDepth(targetDepth uint8) sarray[te[K, V]] {
	if targetDepth < 1 {
		// I prefer to have the trie have at least 2 entries
		panic("targetDepth < 1")
	}

	var resizing *resizingState[te[K, V]]
	for {
		// Is resizing already in progress?
		friendResizing := m.resizing.Load()
		if friendResizing != nil {
			// Wait until the resizer has allocated the new trie so that we can
			// check if that trie is big enough for us.
			friendResizing.allocated.Wait()

			trie := m.newTrie.Load() // NOTE: m.newTrie may be fresher than resizing.newTrie if the resizing already finished and new one started
			globalDepth := (64 - trie.shift) % 64
			if uintptr(targetDepth) > globalDepth {
				// Still too small. Wait until resizing is done and then try getting a chance
				// to resize to our desired size
				m.helpWithCompletingResizing(friendResizing)
				friendResizing.allDone.Wait()
				continue
			}

			m.helpWithCompletingResizing(friendResizing)
			return trie
		}

		// Try to take it for ourselves
		if resizing == nil {
			resizing = new(resizingState[te[K, V]])
			resizing.allDone.Add(1)
			resizing.allocated.Add(1)
		}
		if m.resizing.CompareAndSwap(nil, resizing) {
			// Got it! The actual work is done outside the loop
			break
		}
	}

	// We are now the master of this resizing operation. A swift execution is in place!

	// refresh the current view of the trie
	trie := m.newTrie.Load()
	globalDepth := (64 - trie.shift) % 64

	// re-check the condition in case someone resized the trie while we were waiting
	if uintptr(targetDepth) > globalDepth {
		// Allocate the new trie.
		oldTrie := trie.Slice()
		newTrie := make([]te[K, V], 1<<targetDepth)

		// Special case of empty trie
		if len(oldTrie) == 0 {
			if len(newTrie) != 2 {
				panic("bad")
			}
			bucket := newConBucketMap2[K, V]()
			newTrie[0].Store2(bucket, uint64(bucket.depth))
			newTrie[1].Store2(bucket, uint64(bucket.depth))
		}

		resizing.oldTrie = trie
		resizing.newTrie = asSArray(newTrie)

		// Tell the others about it
		m.newTrie.Update(resizing.newTrie)
		resizing.allocated.Done()

		// For yuge tries we should probably spawn a few helper goroutines to ensure
		// that more than 1 thread works on it.
		if len(newTrie) >= (64 * resizingChunkSize) {
			maxGoros := runtime.GOMAXPROCS(-1)
			numGoros := len(newTrie) / (32 * resizingChunkSize)
			numGoros = min(numGoros, maxGoros) - 1
			for range numGoros {
				go m.helpWithCompletingResizing(resizing)
			}
		}

		// I'm doing my part!
		m.helpWithCompletingResizing(resizing)

		trie = asSArray(newTrie)
		globalDepth = (64 - trie.shift) % 64
		return trie
	}

	// Nothing to do... But we do want to avoid any goroutine from trying to help us do nothing
	resizing.newTrie = trie
	resizing.oldTrie = trie
	resizing.nextChunk.Add(uint64(trie.Len()))
	if !m.resizing.CompareAndSwap(resizing, nil) {
		panic("bad")
	}
	resizing.done.Store(true)
	resizing.allDone.Done()

	// goroutines wait for allocated.Wait() to return before helping. So releasing it
	// last allows us to safely set the other fields to a finished state.
	resizing.allocated.Done()

	return trie
}

func (m *ConMap2[K, V]) helpWithCompletingResizing(r *resizingState[te[K, V]]) {
	if r.done.Load() { // exit quickly for latecomers
		return
	}

	r.allocated.Wait() // caller should already have done this but whatevs

	// We are one of the helpers now
	r.helpers.Add(1)

	oldTrie := r.oldTrie.Slice()
	newTrie := r.newTrie.Slice()
	for {
		start := r.nextChunk.Add(resizingChunkSize) - resizingChunkSize
		if start >= uint64(len(oldTrie)) {
			// If we are the last helper then it's our duty to publish the new trie
			if r.helpers.Add(-1) == 0 && r.done.CompareAndSwap(false, true) {
				m.oldTrie.Update(r.newTrie)

				if !m.resizing.CompareAndSwap(r, nil) {
					panic("bad")
				}
				r.allDone.Done()
			}
			return
		}

		endExclusive := start + resizingChunkSize
		endExclusive = min(endExclusive, uint64(len(oldTrie)))

		// have: [b0 b0 b1 b2]
		// want: [b0 b0 b0 b0 b1 b1 b2 b2]
		oldChunk := asSArray(oldTrie[start:endExclusive])
		newChunk := newTrie[start*2 : endExclusive*2]
		for i := range oldTrie[start:endExclusive] {
			// TODO: Unroll the loop a little and do some non-temporal prefetching
			trieEntry := conTrieGetPacked2(oldChunk, uintptr(i))

			// NOTE: Goroutine doing a split() in the background could
			// try to Store() entries to this new trie. In that case
			// they have the freshest value and thus should win and
			// we should lose. So use CompareAndSwap expecting old value
			// to be nil/zero.
			dst := (*[2]te[K, V])(newChunk[i*2:])
			dst[0].SwapWithPackedIfNil(trieEntry)
			dst[1].SwapWithPackedIfNil(trieEntry)
		}
	}
}

func (m *ConMap2[K, V]) iterateMaps(iter func(*conBucketMap2[K, V])) {
	// Oh... god... modifying iterators... pain...
	trie := m.getTrieForIterating()
	if trie.IsEmpty() {
		return
	}

	slice := trie.Slice()
	globalDepth := (64 - trie.shift) % 64
	for i := 0; i < len(slice); {
		triehash := i
		bucket, _ := slice[triehash].Load2()

		i += 1 << (globalDepth - uintptr(bucket.depth%64))

		iter(bucket)
	}
}

func (m *ConMap2[K, V]) Len() int {
	return int(m.len.Load())
}

func (m *ConMap2[K, V]) getTrieForIterating() sarray[te[K, V]] {
	trie := m.newTrie.Load()
	// If resizing is in progress for this trie then we want to avoid
	// handing a trie with nils to the user.
	if resizing := m.resizing.Load(); resizing != nil {
		resizing.allocated.Wait()
		if trie == resizing.newTrie {
			m.helpWithCompletingResizing(resizing)
			resizing.allDone.Wait()
		}
	}
	return trie
}

// Not allowed to modify the map at all.
func (m *ConMap2[K, V]) All(yield func(K, V) bool) {
	trie := m.getTrieForIterating()

	slice := trie.Slice()

	startPos := uintptr(runtime_fastrand64())
	bucketRotation := uint((startPos >> 32)) % bucketSize
	trieMask := uintptr(len(slice) - 1)
	globalDepth := (64 - trie.shift) % 64

	// align startPos so that we aren't in the middle of a bucket-repeat. This
	// ensures that inside the loop we can always jump forward past the repeats
	// to the next bucket
	{
		trieEntry, _ := slice[startPos&trieMask].Load2()
		startPos &^= (1 << (globalDepth - uintptr(trieEntry.depth))) - 1
	}

	for i := 0; i < len(slice); {
		triehash := startPos + uintptr(i)
		trieEntry, ts := slice[triehash&trieMask].Load2()
		// trieEntry.mu.Lock()

		// did the bucket split when we were locking it?
		if ts != uint64(trieEntry.depth) {
			panic("what to do?")
		}

		i += 1 << ((globalDepth - uintptr(trieEntry.depth)) % 64)

		// TODO: prefetch the next bucket
		// runtime.Prefetch(uintptr(slice[(startPos+uintptr(i))&trieMask].ptr))

		for bucket := trieEntry; bucket != nil; {
			presents := bucket.present.AsMatchiter()

			// Rotate to random starting point
			// TODO: Make every bucket have a different rotation?
			presents = presents.RotateRight(int(bucketRotation))

			kvs := bucket.Kvs()
			for ; presents.HasCurrent(); presents.Advance() {
				slotInBucket := (bucketRotation + uint(presents.Current())) % bucketSize
				kv := &kvs[slotInBucket]
				if ok := yield(kv.Key, kv.Value); !ok {
					// trieEntry.mu.Unlock()
					return
				}
			}

			bucket = bucket.overflow
		}
		// trieEntry.mu.Unlock()
	}
}

// empties the map without releasing the main chunks of memory.
//
// WARNING: Not safe for concurrent use.
func (m *ConMap2[K, V]) Clear() {
	m.iterateMaps(func(bucket *conBucketMap2[K, V]) {
		// zero the KVs, necessary for GC. Because we tend to have a high load-factor just clearing
		// the whole memory region is faster than iterating through all the present
		// slots and individually zeroing them.
		clear(bucket.Kvs())

		// mark all slots as empty
		bucket.present = 0

		// Overflow buckets we just set to nil and free them.
		bucket.overflow = nil
	})

	m.seed = uintptr(runtime_fastrand64())
	m.len.Store(0)
}

// WARNING: Not safe for concurrent use.
func (m *ConMap2[K, V]) Reserve(space int) {
	if m.Len() > 0 {
		panic("reserve called after first Put")
	}

	// NOTE: We seem to get a load factor of ~0.66 basically at any size.
	//
	// The strategy here is to allocate the trie into the full expected capacity
	// but only pre-allocate half of the expected maps to accommodate for the fact
	// that entries are not distributed perfectly evenly.

	adjusted := (space * 7 / 4) / bucketSize

	triesize := max(2, 1<<bits.Len(uint(adjusted)))

	newTrie := make([]te[K, V], triesize)

	// then create the maps
	globalDepth := uint8(bits.Len(uint(triesize)))
	mapdepth := uint8(max(1, globalDepth-2))

	maps := 1 << (mapdepth - 1)
	repeats := 1 << (globalDepth - mapdepth)
	for i := range maps {
		bucket := newConBucketMap2[K, V]() // TODO: Bulk allocate
		bucket.depth = mapdepth - 1

		for j := range repeats {
			newTrie[i*repeats+j].Store2(bucket, uint64(bucket.depth))
		}
	}

	m.oldTrie.Update(asSArray(newTrie))
	m.newTrie.Update(asSArray(newTrie))
}
