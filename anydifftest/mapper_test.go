package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

func TestMapper(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		m := anyvec.MapMax(v.Output(), 3)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Map(m, v)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestMapperTranspose(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		m := anyvec.MapMax(v.Output(), 3)
		myVar := makeRandomVec(c, 18/3)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.MapTranspose(m, myVar)
			},
			V: []*anydiff.Var{myVar},
		}
		ch.FullCheck(t)
	})
}
