"""
    @sum(expr)

a macro that can be used to sum over all the variables in an expression. a faster alternative to `sum(@. expr)`

## Example
```julia
x = [1,2,3]
y = [0.1,0.2,0.3]
z = 2

x1 = @sum(x[i]+y[i]*z)
x2 = sum(@. x+y*z)
x1 ≈ x2 #true
```
"""
macro sum(expr)
    variable_names = Expr(:tuple)
    iterator = Symbol[]
    length_indicator = Symbol[]
    cache = (variable_names.args,iterator,length_indicator)
    if expr.head == :call
        args = expr.args
        for i in 2:length(args)
            __sum_add_variables(cache,args[i])
        end
    else

    end
    iterator = unique!(iterator)
    length(iterator) != 1 && error("@sum: only one iterator index is allowed")
    length(length_indicator) == 0 && error("@sum: no length indicator found")
    res = gensym(:res)
    idx = iterator[1]
    len = length_indicator[1]
    res_expr = Expr(:call,:(Base.promote_eltype))
    append!(res_expr.args,variable_names.args)
    return quote
        let $res = zero($res_expr)
            @inbounds @simd for $idx in 1:first(size($len))
                $res += $expr
            end
        $res
        end
    end  |> esc
end

__sum_add_variables(cache,expr::Number) = nothing

function __sum_add_variables(cache,expr::Symbol)
    vars,_,_ = cache
    push!(vars,expr)
end

function __sum_add_variables(cache,expr::Expr)
    vars,idx,len = cache
    if expr.head == :ref #vector or array
        sym_name = expr.args[1]
        push!(len,sym_name)
        push!(vars,sym_name)
        for j in 2:length(expr.args)
            push!(idx,expr.args[j])
        end
    elseif expr.head == :call
        for i in 2:length(expr.args)
            __sum_add_variables(cache,expr.args[i])
        end     
    end
end
