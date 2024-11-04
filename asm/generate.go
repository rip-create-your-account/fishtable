//go:build ignore

package main

//go:generate go run generate.go -out add.s -stubs stub.go

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
)

func main() {
	// TODO: arm64
	ConstraintExpr("amd64")

	{
		TEXT("FindMatches", NOSPLIT, "func(fingerprints *[16]uint8, fingerprint uint64) (bitmask uint32)")
		Pragma("noescape") // fingerprints have no need to escape

		b1 := Mem{Base: Load(Param("fingerprints"), GP64())}

		xfingerprints := XMM()
		MOVOU(b1, xfingerprints)

		xfingerprint := XMM()
		VPBROADCASTB(must(Param("fingerprint").Resolve()).Addr, xfingerprint)

		PCMPEQB(xfingerprints, xfingerprint)

		hashmatches := GP32()
		PMOVMSKB(xfingerprint, hashmatches)

		Store(hashmatches.As32(), ReturnIndex(0))
		RET()
	}

	Generate()
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}
