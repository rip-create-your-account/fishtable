//go:build amd64 && !nosimd

package fishtable

import (
	"github.com/rip-create-your-account/fishtable/asm"
)

type (
	fingerprintProbe uint64
)

func makeHashProbe(fingerprint fingerprint) fingerprintProbe {
	// Imagine building the 128-bit XMM register here.
	//     MOVD     tophash8, X_tophash8
	//     PSHUFB   X_ZERO, X_tophash8
	return fingerprintProbe(fingerprint)
}

type bucketFinder struct{ fingerprints *[bucketSize]byte }

func bucketFinderFrom(fingerprints *[bucketSize]byte) bucketFinder {
	return bucketFinder{fingerprints}
}

func (b *bucketFinder) ProbeHashMatches(probe fingerprintProbe) (hashes matchiter) {
	hashes.hashMatches = asm.FindMatches(b.fingerprints, uint64(probe))
	return
}
