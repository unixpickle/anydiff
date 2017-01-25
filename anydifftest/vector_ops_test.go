package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

func TestScale(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 15)
		scaler := c.MakeNumeric(-1.5)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Scale(v, scaler)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestAddScaler(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 15)
		scaler := c.MakeNumeric(-1.5)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.AddScaler(v, scaler)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestAdd(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v1 := makeRandomVec(c, 15)
		v2 := makeRandomVec(c, 15)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Add(v1, v2)
			},
			V: []*anydiff.Var{v1, v2},
		}
		ch.FullCheck(t)
	})
}

func TestSub(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v1 := makeRandomVec(c, 15)
		v2 := makeRandomVec(c, 15)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Sub(v1, v2)
			},
			V: []*anydiff.Var{v1, v2},
		}
		ch.FullCheck(t)
	})
}

func TestMul(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v1 := makeRandomVec(c, 15)
		v2 := makeRandomVec(c, 15)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Mul(v1, v2)
			},
			V: []*anydiff.Var{v1, v2},
		}
		ch.FullCheck(t)
	})
}

func TestDiv(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v1 := makeDivisionFriendlyVec(c, 15)
		v2 := makeDivisionFriendlyVec(c, 15)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Div(v1, v2)
			},
			V: []*anydiff.Var{v1, v2},
		}
		ch.FullCheck(t)
	})
}
