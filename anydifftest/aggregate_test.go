package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

func TestSeqSum(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anyseq.Sum(inSeq)
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}
