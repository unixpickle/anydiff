package anydifftest

import (
	"fmt"
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestTanh(t *testing.T) {
	testMathFunction(t, anydiff.Tanh)
}

func TestSigmoidOut(t *testing.T) {
	inVec := anyvec32.MakeVectorData([]float32{1000, -1000, 2, -2, 0})
	inRes := anydiff.NewConst(inVec)
	actual := anydiff.Sigmoid(inRes).Output().Data().([]float32)
	expected := []float32{1, 0, 0.880797078, 0.119202922, 0.5}
	for i, x := range expected {
		a := actual[i]
		if math.IsNaN(float64(a)) || math.Abs(float64(x-a)) > 1e-3 {
			t.Errorf("expected %f but got %f", x, a)
		}
	}
}

func TestSigmoidProp(t *testing.T) {
	testMathFunction(t, anydiff.Sigmoid)
}

func TestLogSoftmax(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.LogSoftmax(v, 6)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestSquare(t *testing.T) {
	testMathFunction(t, anydiff.Square)
}

func TestPowOut(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := anydiff.NewVar(c.MakeVector(15))
		v.Vector.SetData(c.MakeNumericList([]float64{1, 2, 3}))
		actual := getComponents(anydiff.Pow(v, c.MakeNumeric(1.3)).Output())
		expected := []float64{1, 2.4622888267, 4.1711675109}

		for i, x := range expected {
			a := actual[i]
			if math.IsNaN(a) || math.Abs(x-a) > prec {
				t.Errorf("index %d: expected %f but got %f", i, x, a)
			}
		}
	})
}

func TestPowProp(t *testing.T) {
	for _, power := range []float64{1, 2, -2, 0.5} {
		t.Run(fmt.Sprintf("Pow%f", power), func(t *testing.T) {
			runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
				v := anydiff.NewVar(c.MakeVector(15))

				// Avoid numerical issues for finite differences.
				anyvec.Rand(v.Vector, anyvec.Uniform, nil)
				if power != 0.5 {
					if power < 0 {
						v.Vector.AddScaler(c.MakeNumeric(0.25))
					} else {
						v.Vector.Scale(c.MakeNumeric(2))
						v.Vector.AddScaler(c.MakeNumeric(-1))
					}
				}

				powNum := c.MakeNumeric(power)
				ch := &ResChecker{
					F: func() anydiff.Res {
						return anydiff.Pow(v, powNum)
					},
					V: []*anydiff.Var{v},
				}
				ch.FullCheck(t)
			})
		})
	}
}

func TestClipPos(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := anydiff.NewVar(c.MakeVector(15))

		// Avoid numerical issues for finite differences.
		anyvec.Rand(v.Vector, anyvec.Uniform, nil)
		v.Vector.AddScaler(c.MakeNumeric(0.1))
		mask := c.MakeVector(15)
		anyvec.Rand(mask, anyvec.Bernoulli, nil)
		mask.AddScaler(c.MakeNumeric(-0.5))
		v.Vector.Mul(mask)

		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.ClipPos(v)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func testMathFunction(t *testing.T, f func(v anydiff.Res) anydiff.Res) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return f(v)
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}
