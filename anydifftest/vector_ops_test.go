package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

func TestScale(t *testing.T) {
	testScalerOp(t, anydiff.Scale)
}

func TestAddScalar(t *testing.T) {
	testScalerOp(t, anydiff.AddScalar)
}

func testScalerOp(t *testing.T, f func(v anydiff.Res, s anyvec.Numeric) anydiff.Res) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 15)
		scaler := c.MakeNumeric(-1.5)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return f(v, scaler)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestAdd(t *testing.T) {
	testComponentwiseOp(t, anydiff.Add)
}

func TestSub(t *testing.T) {
	testComponentwiseOp(t, anydiff.Sub)
}

func TestMul(t *testing.T) {
	testComponentwiseOp(t, anydiff.Mul)
}

func testComponentwiseOp(t *testing.T, f func(v1, v2 anydiff.Res) anydiff.Res) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v1 := makeRandomVec(c, 15)
		v2 := makeRandomVec(c, 15)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return f(v1, v2)
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
