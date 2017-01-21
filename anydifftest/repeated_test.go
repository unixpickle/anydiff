package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

func TestAddRepeated(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		biases := makeRandomVec(c, 6)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.AddRepeated(v, biases)
			},
			V: []*anydiff.Var{v, biases},
		}
		ch.FullCheck(t)
	})
}

func TestScaleRepeated(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		scalers := makeRandomVec(c, 6)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.ScaleRepeated(v, scalers)
			},
			V: []*anydiff.Var{v, scalers},
		}
		ch.FullCheck(t)
	})
}
