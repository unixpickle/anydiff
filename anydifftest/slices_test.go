package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

func TestSliceVar(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Slice(v, 5, 10)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestSliceRes(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Slice(anydiff.Exp(v), 5, 10)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}
