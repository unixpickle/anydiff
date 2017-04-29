package anyfwd

import "github.com/unixpickle/anyvec"

// NumOps implements anyvec.NumOps for Numeric values.
type NumOps struct {
	Creator *Creator
}

// Add adds two Numerics.
func (n NumOps) Add(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	sum := Numeric{
		Value: n.valueOps().Add(num1.Value, num2.Value),
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		sum.Grad = append(sum.Grad, n.valueOps().Add(g1, g2))
	}
	return sum
}

// Sub subtracts two Numerics.
func (n NumOps) Sub(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	sum := Numeric{
		Value: n.valueOps().Sub(num1.Value, num2.Value),
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		sum.Grad = append(sum.Grad, n.valueOps().Sub(g1, g2))
	}
	return sum
}

// Mul multiplies two Numerics.
func (n NumOps) Mul(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	sum := Numeric{
		Value: n.valueOps().Mul(num1.Value, num2.Value),
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		productRule := n.valueOps().Add(
			n.valueOps().Mul(g1, num2.Value),
			n.valueOps().Mul(num1.Value, g2),
		)
		sum.Grad = append(sum.Grad, productRule)
	}
	return sum
}

// Div divides two Numerics.
func (n NumOps) Div(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	sum := Numeric{
		Value: n.valueOps().Div(num1.Value, num2.Value),
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		quotientRule := n.valueOps().Sub(
			n.valueOps().Div(g1, num2.Value),
			n.valueOps().Div(
				n.valueOps().Mul(num1.Value, g2),
				n.valueOps().Mul(num2.Value, num2.Value),
			),
		)
		sum.Grad = append(sum.Grad, quotientRule)
	}
	return sum
}

// Pow raises n1 to the n2 power.
//
// This only works if n2 is a constant.
func (n NumOps) Pow(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	if !n.Creator.constant(num2) {
		panic("exponent must be constant")
	}

	ops := n.valueOps()
	c := n.Creator.ValueCreator
	expMinusOne := ops.Sub(num2.Value, c.MakeNumeric(1))

	res := Numeric{
		Value: ops.Pow(num1.Value, num2.Value),
		Grad:  make([]anyvec.Numeric, len(num1.Grad)),
	}

	deriv := ops.Mul(num2.Value, ops.Pow(num1.Value, expMinusOne))
	for i, baseGrad := range num1.Grad {
		res.Grad[i] = ops.Mul(baseGrad, deriv)
	}

	return res
}

// Identical checks if two Numerics are exactly identical,
// including their derivatives.
func (n NumOps) Identical(n1, n2 anyvec.Numeric) bool {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	if !n.valueOps().Identical(num1.Value, num2.Value) {
		return false
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		if !n.valueOps().Identical(g1, g2) {
			return false
		}
	}
	return true
}

// Equal checks if two Numerics have the same value.
func (n NumOps) Equal(n1, n2 anyvec.Numeric) bool {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	return n.valueOps().Equal(num1.Value, num2.Value)
}

// Less checks if one Numeric is less than another.
func (n NumOps) Less(n1, n2 anyvec.Numeric) bool {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	return n.valueOps().Less(num1.Value, num2.Value)
}

// Greater checks if one Numeric is greater than another.
func (n NumOps) Greater(n1, n2 anyvec.Numeric) bool {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	return n.valueOps().Greater(num1.Value, num2.Value)
}

func (n NumOps) valueOps() anyvec.NumOps {
	return n.Creator.ValueCreator.NumOps()
}
