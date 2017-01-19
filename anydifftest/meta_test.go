package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

func TestPool(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Pool(anydiff.Tanh(v), func(r anydiff.Res) anydiff.Res {
					return anydiff.Mul(r, r)
				})
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}
