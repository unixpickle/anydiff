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

func TestConcat(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v1 := makeRandomVec(c, 18)
		v2 := makeRandomVec(c, 5)
		v3 := makeRandomVec(c, 8)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Concat(v1, anydiff.Exp(v2), anydiff.Sin(v3))
			},
			V: []*anydiff.Var{v1, v2, v3},
		}
		ch.FullCheck(t)
	})
}
