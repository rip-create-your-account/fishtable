// Code generated by command: go run generate.go -out add.s -stubs stub.go. DO NOT EDIT.

//go:build amd64

#include "textflag.h"

// func FindMatches(fingerprints *[16]uint8, fingerprint uint64) (bitmask uint32)
// Requires: AVX2, SSE2
TEXT ·FindMatches(SB), NOSPLIT, $0-20
	MOVQ         fingerprints+0(FP), AX
	MOVOU        (AX), X0
	VPBROADCASTB fingerprint+8(FP), X1
	PCMPEQB      X0, X1
	PMOVMSKB     X1, AX
	MOVL         AX, bitmask+16(FP)
	RET
