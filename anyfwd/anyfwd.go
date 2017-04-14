// Package anyfwd is a forward automatic-differentiation
// plug-in for anyvec.
// It wraps an anyvec.Creator to implement a dual number
// system.
package anyfwd

import "github.com/unixpickle/anyvec"

const badJacobianErr = "bad jacobian size"

// Numeric is a dual number.
// It includes both a value and a gradient of that value.
//
// All Numeric instances for a given Creator should have
// the same number of gradient components.
type Numeric struct {
	Value anyvec.Numeric
	Grad  []anyvec.Numeric
}

// NumericList is a dual Vector.
type NumericList struct {
	Values anyvec.NumericList

	// Jacobian stores the derivative of Values with respect
	// to each variable.
	//
	// Each Jacobian entry is a separate object, e.g. a
	// different slice with a different backing array.
	// This ensures that modifying one jacobian Vector
	// will not modify another one.
	Jacobian []anyvec.NumericList
}

// A Creator is an anyvec.Creator for dual Vectors.
type Creator struct {
	// ValueCreator is the underlying Creator used to
	// deal with the sub-parts of dual Vectors.
	ValueCreator anyvec.Creator

	// GradSize is the number of gradient components.
	GradSize int
}

// MakeNumeric creates a Numeric with a zero gradient.
func (c *Creator) MakeNumeric(x float64) anyvec.Numeric {
	res := Numeric{
		Value: c.ValueCreator.MakeNumeric(x),
	}
	for i := 0; i < c.GradSize; i++ {
		res.Grad = append(res.Grad, c.ValueCreator.MakeNumeric(0))
	}
	return res
}

// MakeNumericList creates a NumericList with zero
// gradients.
func (c *Creator) MakeNumericList(x []float64) anyvec.NumericList {
	res := NumericList{Values: c.ValueCreator.MakeNumericList(x)}
	zeros := make([]float64, len(x))
	for i := 0; i < c.GradSize; i++ {
		res.Jacobian = append(res.Jacobian, c.ValueCreator.MakeNumericList(zeros))
	}
	return res
}

// MakeVector creates a zero anyvec.Vector.
func (c *Creator) MakeVector(size int) anyvec.Vector {
	res := &Vector{
		Values: c.ValueCreator.MakeVector(size),
	}
	for i := 0; i < c.GradSize; i++ {
		res.Jacobian = append(res.Jacobian, c.ValueCreator.MakeVector(size))
	}
	return res
}

// MakeVectorData creates an anyvec.Vector from the
// NumericList.
func (c *Creator) MakeVectorData(data anyvec.NumericList) anyvec.Vector {
	nl := data.(NumericList)
	res := &Vector{
		Values: c.ValueCreator.MakeVectorData(nl.Values),
	}
	if len(nl.Jacobian) != c.GradSize {
		panic(badJacobianErr)
	}
	for _, x := range nl.Jacobian {
		res.Jacobian = append(res.Jacobian, c.ValueCreator.MakeVectorData(x))
	}
	return res
}

// Concat concatenates the Vectors.
func (c *Creator) Concat(vs ...anyvec.Vector) anyvec.Vector {
	valVecs := make([]anyvec.Vector, len(vs))
	jacobianVecs := make([][]anyvec.Vector, c.GradSize)
	for i, v := range vs {
		vec := v.(*Vector)
		valVecs[i] = vec.Values
		if len(vec.Jacobian) != c.GradSize {
			panic(badJacobianErr)
		}
		for j, grad := range vec.Jacobian {
			jacobianVecs[j] = append(jacobianVecs[j], grad)
		}
	}
	res := &Vector{Values: c.ValueCreator.Concat(valVecs...)}
	for _, grad := range jacobianVecs {
		res.Jacobian = append(res.Jacobian, c.ValueCreator.Concat(grad...))
	}
	return res
}

// MakeMapper creates a Mapper based on the lookup table.
func (c *Creator) MakeMapper(inSize int, table []int) anyvec.Mapper {
	return &Mapper{
		CreatorPtr:  c,
		ValueMapper: c.ValueCreator.MakeMapper(inSize, table),
	}
}
